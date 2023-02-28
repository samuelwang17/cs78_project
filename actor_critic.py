# total actor critic package, encapsulating a single worker/env pair

import torch
import torch.nn as nn
from transformer import mod_transformer as RLformer
from tokenizer import Tokenizer
from pokerenv import poker_env
from itertools import chain
import numpy as np
from model_components import grad_skip_softmax, grad_skip_logsoftmax
import time


class Agent(nn.Module):
    def __init__(self,
                 model_dim,
                 mlp_dim,
                 attn_heads,
                 sequence_length,
                 enc_layers,
                 memory_layers,
                 mem_length,
                 dec_layers,
                 action_dim,
                 ) -> None:
        super().__init__()
        self.model = RLformer(
            model_dim=model_dim,
            mlp_dim=mlp_dim,
            attn_heads=attn_heads,
            sequence_length=sequence_length,
            enc_layers=enc_layers,
            memory_layers=memory_layers,
            mem_length=mem_length,
            dec_layers=dec_layers,
            action_dim=action_dim,
        )
        self.tokenizer = Tokenizer(model_dim=model_dim)
        self.hand_dict = {}

    def init_player(self, player, hand):
        # initialize this players hand and tokenize it, store it in buffer
        hand_tensor = self.tokenizer(hand)  # hand needs to be card observations -- list of length two of tensors
        assert hand_tensor != None
        self.hand_dict[player] = hand_tensor

    def forward(self, player, obs_flat, games_per_run, new_hand):
        # takes flattened inputs in list form, not tokenized
        enc_input = []
        dec_input = []
        for index, obs in enumerate(obs_flat):
            dec_input_temp = []
            enc_input_temp = []
            for x in range(len(obs)):  # expects obs_flat to be a list of flattened observation lists
                dec_input_temp.append(self.tokenizer(obs[x]))  # env steam is what obs_flat used to be
                enc_input_temp.append(self.hand_dict[player[index]])
            dec_input.append(torch.stack(dec_input_temp).squeeze())
            enc_input.append(torch.stack(enc_input_temp).squeeze())
        policy_logits, value = self.model(enc_input, dec_input, player, new_hand, games_per_run)  # expects a list of tensors for dec_input
        return policy_logits, value


class actor_critic():
    # Needs to be able to run hand, return loss with grad enabled
    def __init__(self,
                 model_dim,
                 mlp_dim,
                 heads,
                 enc_layers,
                 memory_layers,
                 mem_length,
                 dec_layers,
                 max_sequence: int = 200,
                 n_players: int = 2,
                 gamma: float = .8,
                 n_actions: int = 10,  # random placeholder value
                 games_per_run: int=128,
                 ) -> None:

        self.gamma = gamma
        self.env = poker_env(n_players=n_players, batch_size=games_per_run)
        self.agent = Agent(
            model_dim=model_dim,
            mlp_dim=mlp_dim,
            attn_heads=heads,
            sequence_length=max_sequence,
            enc_layers=enc_layers,
            memory_layers=memory_layers,
            mem_length=mem_length,
            dec_layers=dec_layers,
            action_dim=n_actions,
        )
        self.n_players = n_players
        self.games_per_run = games_per_run
        self.observations = [[] for _ in
                             range(games_per_run)]  # this will be a list of lists, each is the list of observations in a hand
        self.obs_flat = [[] for _ in range(self.games_per_run)]

        self.rewards = [[] for _ in range(self.games_per_run)]
        self.rewards_flat = [[] for _ in range(self.games_per_run)]

        self.values = [[] for _ in range(self.games_per_run)]
        self.val_flat = [[] for _ in range(self.games_per_run)]

        self.action_log_probabilies = [[] for _ in range(self.games_per_run)]
        self.alp_flat = [[] for _ in range(self.games_per_run)]

        self.max_sequence = max_sequence

        self.n_players = n_players

        self.n_actions = n_actions

        self.softmax = nn.Softmax()

        self.padding = nn.ConstantPad1d(
            padding=n_players - 1,
            value=-100000
        )

        self.lsm = grad_skip_logsoftmax()

        self.silu = nn.SiLU()

        self.time_dict = {'total': 0, 'env': 0, 'model_inference': 0, 'loss': 0}

    def sample_action(self, curr_logits, batch_num):
        # MASK, SAMPLE, DETOKENIZE
        # get the player about to act
        player = self.env.in_turn[batch_num]
        player_stack = self.env.stacks[batch_num][player]
        pot = self.env.pot[batch_num]
        linspace_dim = self.n_actions - 4
        assert linspace_dim > 0

        # MASK
        mask = [0] * self.n_actions
        if self.env.behind[player] != 0:
            mask[3] = 1  # cannot check
        else:
            mask[1] = 1  # cannot call
            mask[2] = 1  # cannot fold if not facing a bet

        if self.env.behind[player] == self.env.stacks[player]:
            mask[0] = 1  # cannot shove if shoving and calling are equivalent

        linspace = np.geomspace(.5, 2, num=linspace_dim)
        stack_checker = lambda x: 1 if x * pot >= player_stack or x * pot < (2 * self.env.current_largest_bet[batch_num]) or x * pot < self.env.behind[batch_num][player] else 0
        mask[4:] += np.fromiter((stack_checker(x) for x in linspace),linspace.dtype)  # mask away bets that are larger than stack or all in or bets that are are not 2x larger than last bet
        tensor_mask = torch.Tensor(mask)

        # grad skip softmax -- neural replicator dynamics, with mask
        policy = self.softmax(curr_logits.masked_fill(tensor_mask == 1, float('-inf')))

        np_dist = policy.detach().numpy()

        # SAMPLE
        action_index = np.random.choice(self.n_actions, p=np_dist)
        # calculate action log prob for use in advantage later
        # y_logit = curr_logits[-1][action_index] #.0001 used to avoid log(0) causing grad issues
        alps = self.lsm(curr_logits)
        alp = alps[action_index]

        # DETOKENIZE
        if action_index == 0:  # all in
            action = {'player': player, 'type': 'bet', 'value': player_stack}
        elif action_index == 1:  # call
            action = {'player': player, 'type': 'call', 'value': 0}
        elif action_index == 2:  # fold
            action = {'player': player, 'type': 'fold', 'value': 0}
        elif action_index == 3:  # check
            action = {'player': player, 'type': 'bet', 'value': 0}
        else:
            action = {'player': player, 'type': 'bet', 'value': (linspace[action_index - 4] * pot) // 1}

        return alp, action

    def init_hands(self):
        # get all hands
        # run encoder for each of players
        for player in range(self.n_players):
            hand = self.env.get_hand(player)
            self.agent.init_player(player, hand)

    def chop_seq(self):
        # if length of observations is above a certain size, chop it back down to under sequence length by removing oldest hand
        # return flattened version to give to model on next run
        self.obs_flat = [[] for i in range(self.games_per_run)]
        for index, observation in enumerate(self.observations):
            self.obs_flat[index].append(list(chain(*observation)))
        for i in range(len(self.observations)):
            while len(self.obs_flat[i][0]) > self.max_sequence:
                self.observations[i] = self.observations[i][1:]
                self.obs_flat[i][0] = list(chain(*self.observations[i]))

                self.rewards[i] = self.rewards[i][1:]
                self.rewards_flat[i] = list(chain(*self.rewards[i]))

                self.values[i] = self.values[i][1:]
                self.val_flat[i] = list(chain(*self.values[i]))

                self.action_log_probabilies[i] = self.action_log_probabilies[i][1:]
                self.alp_flat[i] = list(chain(*self.action_log_probabilies[i]))


            else:
                self.rewards_flat[i] = list(chain(*self.rewards[i]))
                self.val_flat[i] = list(chain(*self.values[i]))
                self.alp_flat[i] = list(chain(*self.action_log_probabilies[i]))

        assert len(self.obs_flat[i]) <= self.max_sequence

    def play_hand(self):
        hand_over = [False for i in range(self.games_per_run)]
        self.time_dict = {'total': 0, 'env': 0, 'model_inference': 0, 'loss': 0}
        clock1 = time.time_ns()
        # makes agent play one hand
        # deal cards
        rewards, observations = self.env.new_hand() # start a new hand, observations is list of observation lists
        self.init_hands()  # pre load all of the hands
        # init lists for this hand
        for x in range(len(self.observations)):
            self.observations[x].append(observations[x])
        for x in range(len(self.rewards)):
            self.rewards[x].append(rewards[x])
            self.values[x].append([])
            self.action_log_probabilies[x].append([])
        self.chop_seq()  # prepare for input to model
        for x in range(self.games_per_run):
            for y in range(len(rewards[x])):
                new_values = [-5783 for i in range(self.n_players)]  # -5783 filler value
                self.values[x][-1].append(torch.Tensor(new_values))
                new_alps = [0] * self.n_players
                self.action_log_probabilies[x][-1].append(torch.Tensor(new_alps))

        all_over = False
        while not all_over:
            # get values and policy -- should be in list form over sequence length
            player = self.env.in_turn

            clock = time.time_ns()
            policy_logits, values = self.agent(player, self.obs_flat, self.games_per_run, new_hand=True)
            vals = []
            alps = []
            actions = []
            for x in range(len(values)):
                if not hand_over[x]:
                    vals.append(values[x])
                    this_alp, action = self.sample_action(policy_logits[x][-1], x)
                    alps.append(this_alp)
                    actions.append(action)
                else:
                    actions.append(-1) # filler value, shouldn't matter as the environment also keeps track of hand status
                    vals.append('hand_over')
                    alps.append('hand_over')
            r, o, h = self.env.take_actions(actions)
            all_over = True
            for index, x in enumerate(r):
                if not hand_over[index]:
                    all_over = False
                    self.rewards[index][-1] += r[index]  # add tensor
                    self.observations[index][-1] += o[index]

            # value needs to be on a per player basis
            for index, value in enumerate(values):
                if not hand_over[index]:
                    new_value = self.padding(value[-1].unsqueeze(-1))[
                                 (self.n_players - player[index] - 1):(2 * self.n_players - player[index] - 1)].squeeze()
                    self.values[index][-1].append(new_value)
                    diff = len(self.rewards[index][-1]) - len(self.values[index][-1])
                    for x in range(diff):
                        new_values = [-100000 for i in range(self.n_players)]  # -10000 is filler value
                        # fill_tensor = torch.Tensor(new_values)
                        self.values[index][-1].append(torch.Tensor(new_values))

            for index, alp in enumerate(alps):
                if not hand_over[index]:
                    new_alp = self.padding(alp.unsqueeze(-1))[
                              (self.n_players - player[index] - 1):(2 * self.n_players - player[index] - 1)].squeeze()
                    self.action_log_probabilies[index][-1].append(new_alp)
                    diff = len(self.rewards[index][-1]) - len(self.action_log_probabilies[index][-1])
                    for x in range(diff):
                        new_alps = [-100000 for i in range(self.n_players)]
                        self.action_log_probabilies[index][-1].append(torch.Tensor(new_alps))
            for index, bool in enumerate(h):
                hand_over[index] = bool
            # prepare for next action
        self.chop_seq()

        clock = time.time_ns()
        # _, V_T = self.agent(player, self.obs_flat, new_hand=False)
        # Vals_T = [0 for i in range(self.n_players)]
        # Vals_T[player] = V_T.squeeze()[-1]
        Vals_T = torch.zeros([self.games_per_run, 2])
        self.time_dict['model_inference'] += time.time_ns() - clock
        # process gradients and return loss:
        out = self.get_loss(Vals_T)
        self.time_dict['total'] = time.time_ns() - clock1
        return out

    def get_loss(self, Vals_T):
        clock = time.time_ns()
        Q_t = Vals_T # 128 2 zeros
        actor_loss_total = 0
        critic_loss_total = 0
        for i in range(self.games_per_run):
            Qs = [[] for i in range(len(self.rewards[i][-1]))]
            for t in reversed(range(len(self.rewards[i][-1]))):
                Q_t[i] = torch.tensor(self.rewards[i][-1][t]) + self.gamma * Q_t[i] # adds rewards up going backwards to get vals
                Qs[t] = Q_t[i]
            Qs = torch.stack(Qs)[1:]  # 2d tensor sequence, players
            values = torch.stack(self.values[i][-1])[:-1]

            # set Qs to filler value where value is filler value
            Qs = Qs.masked_fill(values == -100000, -100000)
            alps = torch.stack(self.action_log_probabilies[i][-1])[:-1]
            advantages = Qs - values
            advantages = advantages.masked_fill(values == -5783,
                                                0)  # using arbitrary filler from earlier to mask out the blinds

            actor_loss = (-alps * advantages).mean()  # loss function for policy going into softmax on backpass
            actor_loss_total += actor_loss
            critic_loss = (advantages).pow(2).mean() / 200  # autogressive critic loss - MSE
            critic_loss_total += critic_loss
        actor_loss_total = actor_loss_total / self.games_per_run
        critic_loss_total = critic_loss_total / self.games_per_run
        loss = actor_loss_total + critic_loss_total
        self.time_dict['loss'] = time.time_ns() - clock
        return loss, actor_loss, critic_loss, self.time_dict

    def clear_memory(self):
        # zero the gradients of the model
        self.agent.model.zero_grad()