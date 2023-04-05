# total actor critic package, encapsulating a single worker/env pair

import torch
import torch.nn as nn
from transformer import mod_transformer as RLformer
from tokenizer import Tokenizer
from pokerenv import poker_env
from itertools import chain
import numpy as np
from model_components import grad_skip_softmax, grad_skip_logsoftmax, symlog
import time
class Agent(nn.Module):
    def __init__(self,
        model_dim,
        mlp_dim,
        attn_heads,
        sequence_length,
        dec_layers,
        action_dim,
        token_args
    ) -> None:
        super().__init__()
        self.model = RLformer(
            model_dim = model_dim,
            mlp_dim = mlp_dim,
            attn_heads = attn_heads,
            sequence_length = sequence_length,
            dec_layers = dec_layers,
            action_dim = action_dim,
        )
        self.tokenizer =  Tokenizer(token_dim_dict = token_args)
        self.hand_dict = {}

    def forward(self, player, obs_flat):
        #takes flattened inputs in list form, not tokenized
        seq = self.tokenizer(obs_flat[player])
        policy_logits, value = self.model(seq)
        return policy_logits, value

class actor_critic():
    #Needs to be able to run hand, return loss with grad enabled
    def __init__(self, 
    model_dim,
    mlp_dim,
    heads,
    dec_layers,
    token_args,
    max_sequence: int = 200, 
    n_players: int = 2,
    gamma: float = .8,
    n_actions: int = 10, # random placeholder value
    ) -> None:
    
        self.gamma = gamma
        self.env = poker_env(n_players = n_players)
        self.agent = Agent(
            model_dim = model_dim,
            mlp_dim = mlp_dim,
            attn_heads = heads,
            sequence_length = max_sequence,
            dec_layers = dec_layers,
            action_dim = n_actions,
            token_args= token_args
        )
        self.n_players = n_players
        self.observations = {}
        self.obs_flat = {}
        for x in range(n_players):
            self.observations[x] = [] #this will be a list of lists, each is the list of observations in a hand
            self.obs_flat[x] = list(chain(*self.observations[x]))
        
        self.rewards = []
        self.rewards_flat = list(chain(*self.rewards))

        self.values = []
        self.val_flat = list(chain(*self.values))

        self.action_log_probabilies = []
        self.alp_flat = list(chain(*self.action_log_probabilies))

        self.max_sequence = max_sequence

        self.n_players = n_players

        self.n_actions = n_actions

        self.softmax = grad_skip_softmax()

        self.padding = nn.ConstantPad1d(
            padding = n_players - 1,
            value = -100000
        )
        self.padding_alps = nn.ConstantPad1d(
            padding = n_players - 1,
            value = 0
        )
        self.lsm = grad_skip_logsoftmax()

        self.symlog = symlog()

        self.time_dict = {'total': 0, 'env': 0, 'model_inference': 0, 'loss': 0}


    def sample_action(self, curr_logits):
        # MASK, SAMPLE, DETOKENIZE
        # get the player about to act
        player = self.env.in_turn
        player_stack = self.env.stacks[player]
        pot = self.env.pot
        linspace_dim = self.n_actions - 4
        assert linspace_dim > 0
        
        # MASK
        mask = [0] * self.n_actions
        if self.env.behind[player] != 0:
            mask[3] = 1 # cannot check
        else:
            mask[1] = 1 # cannot call
            mask[2] = 1 # cannot fold if not facing a bet

        if self.env.behind[player] == player_stack:
            mask[0] = 1  # cannot shove if shoving and calling are equivalent
        
        #CURRENT QUICK FIX
        # cannot all in if all in is > 2x pot
        if player_stack >  2 * pot:
            mask[0] = 1  # cannot shove if shoving and calling are equivalent




        linspace = np.geomspace(.5, 2, num = linspace_dim)
        stack_checker = lambda x: 1 if x * pot >= player_stack or x * pot < (2 * self.env.current_largest_bet) else 0
        mask[4:] += np.fromiter((stack_checker(x) for x in linspace), linspace.dtype) # mask away bets that are larger than stack or all in or bets that are are not 2x larger than last bet
        tensor_mask = torch.Tensor(mask)

        # grad skip softmax -- neural replicator dynamics, with mask
        policy = self.softmax(curr_logits.masked_fill(tensor_mask == 1, float('-inf')))

        np_dist = np.squeeze(policy[-1].detach().numpy())
        

        # SAMPLE
        action_index = np.random.choice(self.n_actions, p=np_dist)
        # calculate action log prob for use in advantage later
        # y_logit = curr_logits[-1][action_index] #.0001 used to avoid log(0) causing grad issues
        alps = self.lsm(curr_logits)[-1]
        alp = alps[action_index]

        # DETOKENIZE
        if action_index == 0: # all in
            action = {'player': player, 'type': 'bet', 'value': player_stack}
        elif action_index == 1: # call
            action = {'player': player, 'type': 'call', 'value': 0}
        elif action_index == 2: # fold
            action = {'player': player, 'type': 'fold', 'value': 0}
        elif action_index == 3: # check
            action = {'player': player, 'type': 'bet', 'value': 0}
        else:
            action = {'player': player, 'type': 'bet', 'value': (linspace[action_index - 4] * pot) // 1}

        return alp, action
    
    def init_hands(self):
        # get all hands
        for player in range(self.n_players):
            hand = self.env.get_hand(player)
            self.observations[player].append(hand)
    
    def chop_seq(self):
        #if length of observations is above a certain size, chop it back down to under sequence length by removing oldest hand
        #return flattened version to give to model on next run
        for x in range(self.n_players):
            self.obs_flat[x] = list(chain(*self.observations[x]))
        if len(self.obs_flat[0]) > self.max_sequence:
            
            for x in range(self.n_players):
                before = len(self.obs_flat[x])
                self.observations[x] = self.observations[x][1:]
                self.obs_flat[x] = list(chain(*self.observations[x]))
                after = len(self.obs_flat[x])
                assert len(self.obs_flat[x]) <= self.max_sequence, f'prechop: {before}, postchop: {after}'

            self.rewards = self.rewards[1:]
            self.rewards_flat = list(chain(*self.rewards))
            

            self.values = self.values[1:]
            self.val_flat = list(chain(*self.values))

            self.action_log_probabilies = self.action_log_probabilies[1:]
            self.alp_flat = list(chain(*self.action_log_probabilies))
            

        else:
            for x in range(self.n_players):
                self.obs_flat[x] = list(chain(*self.observations[x]))
            self.rewards_flat = list(chain(*self.rewards))
            self.val_flat = list(chain(*self.values))
            self.alp_flat = list(chain(*self.action_log_probabilies))
        
        assert len(self.obs_flat) <= self.max_sequence


    def play_hand(self):
        self.time_dict = {'total': 0, 'env': 0, 'model_inference': 0, 'loss': 0}
        clock1 = time.time_ns()
        # makes agent play one hand
        # deal cards
        rewards, observations = self.env.new_hand() # start a new hand
        self.init_hands() # pre load all of the hands
        # init lists for this hand
        for x in range(self.n_players):
            self.observations[x][-1] += observations
        self.rewards.append(rewards)
        self.chop_seq() # prepare for input to model
        self.values.append([])
        self.action_log_probabilies.append([])
        for x in range(len(rewards)):
                new_values = [-1e5] * self.n_players # -5783 fed here so that 
                self.values[-1].append(torch.Tensor(new_values))
                new_alps = [0] * self.n_players
                self.action_log_probabilies[-1].append(torch.Tensor(new_alps))
        
        hand_over = False
        while not hand_over:
            # get values and policy -- should be in list form over sequence length
            player = self.env.in_turn
            clock = time.time_ns()
            policy_logits, values = self.agent(player, self.obs_flat)
            self.time_dict['model_inference'] += time.time_ns() - clock
            
            value = values.squeeze()[-1] # get last value estimate
            assert value.requires_grad
            curr_logits = policy_logits[-1] # get last policy distribution
            

            alp, action = self.sample_action(curr_logits) # handles mask, softmax, sample, detokenization
            
            rewards, obs, hand_over = self.env.take_action(action) # need to change environment to return hand_over boolean
            
            # add new information from this step
            self.rewards[-1] += rewards #add tensor
            for x in range(self.n_players):
                self.observations[x][-1] += obs

            # value needs to be on a per player basis

            new_values = self.padding(value.unsqueeze(-1))[(self.n_players - player - 1):(2 * self.n_players - player - 1)].squeeze()
            self.values[-1].append(new_values)
            for x in range(len(rewards) - 1):
                new_values = [-1e5] * self.n_players #-10000 is filler value
                # fill_tensor = torch.Tensor(new_values)
                self.values[-1].append(torch.Tensor(new_values))
            
            new_alp = self.padding_alps(alp.unsqueeze(-1))[(self.n_players - player - 1):(2 * self.n_players - player - 1)].squeeze()
            self.action_log_probabilies[-1].append(new_alp)
            for x in range(len(rewards) - 1):
                new_alps = [0] * self.n_players
                self.action_log_probabilies[-1].append(torch.Tensor(new_alps))
                

            # prepare for next action
            self.chop_seq()

        clock = time.time_ns()
        Vals_T = [0] * self.n_players
        for player in range(self.n_players):
            _, V_T = self.agent(player, self.obs_flat)
            Vals_T[player] = V_T.squeeze()[-1]
        self.time_dict['model_inference'] += time.time_ns() - clock
        # process gradients and return loss:
        out = self.get_loss(torch.stack(Vals_T))
        self.time_dict['total'] = time.time_ns() - clock1
        self.env.give_losses(out)
        return out

    def get_loss(self, Vals_T):
        clock = time.time_ns()
        Qs = [0] * len(self.rewards[-1])
        Q_t = torch.Tensor([0] * self.n_players) # expected sum of future rewards is zero at gto, or self play in general
        # first column back
        Q_t = sum(self.rewards[-1]) + self.gamma * Q_t #adds rewards up going backwards to get vals
        Qs[-1] = Q_t
        for t in reversed(range(len(self.rewards[-1]) - 1)):
            Q_t = self.gamma * Q_t #adds rewards up going backwards to get vals
            Qs[t] = Q_t
        

        Qs = torch.stack(Qs)[1:] #2d tensor sequence, players

        values = torch.stack(self.values[-1])[:-1]
        values[-1] = Vals_T

        # set Qs to filler value where value is filler value
        #Qs = Qs.masked_fill(values == -100000, -100000)
        alps = torch.stack(self.action_log_probabilies[-1])[:-1]
        advantages = Qs - values 
        advantages = advantages.masked_fill(values == -1e5, 0) # using arbitrary filler from earlier to mask out the blinds
        advantages = self.symlog(advantages)
        actor_loss_mat = (-alps * advantages)
        critic_loss_mat = (advantages).pow(2)

        actor_loss = actor_loss_mat[:, :1].sum() # loss function for policy going into softmax on backpass
        critic_loss = (critic_loss_mat[:, :1].sum()) # autogressive critic loss - MSE

        loss = actor_loss + critic_loss
        self.time_dict['loss'] = time.time_ns() - clock
        return loss, actor_loss, critic_loss, self.time_dict
    

    def clear_memory(self):
        # zero the gradients of the model
        self.agent.model.zero_grad()
    