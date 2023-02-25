# total actor critic package, encapsulating a single worker/env pair

import torch
import torch.nn as nn
from transformer import mod_transformer as RLformer
from tokenizer import Tokenizer
from pokerenv import poker_env
from itertools import chain
import numpy as np
from model_components import grad_skip_softmax

class Agent(nn.Module):
    def __init__(self,
        model_dim,
        mlp_dim,
        attn_heads,
        sequence_length,
        enc_layers,
        dec_layers,
        action_dim,
    ) -> None:
        super().__init__()
        self.model = RLformer(
            model_dim = model_dim,
            mlp_dim = mlp_dim,
            attn_heads = attn_heads,
            sequence_length = sequence_length,
            enc_layers = enc_layers,
            dec_layers = dec_layers,
            action_dim = action_dim,
        )
        self.tokenizer =  Tokenizer(model_dim=model_dim)

    def init_player(self, player, hand):
        # initialize this players hand and tokenize it, store it in buffer
        hand_tensor = self.tokenizer(hand) #hand needs to be card observations -- list of length two of tensors
        self.register_buffer(f'hand_{player}', tensor= hand_tensor)

    def forward(self, player, obs_flat):
        #takes flattened inputs in list form, not tokenized
        enc_input = self.get_buffer(f'hand_{player}')
        dec_input = self.tokenizer(obs_flat)
        policy_logits, value = self.model(enc_input, dec_input)
        return policy_logits, value

class actor_critic():
    #Needs to be able to run hand, return loss with grad enabled
    def __init__(self, 
    model_dim,
    mlp_dim,
    heads,
    enc_layers,
    dec_layers,
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
            enc_layers = enc_layers,
            dec_layers = dec_layers,
            action_dim = n_actions,
        )
        self.n_players = n_players
        self.observations = [] #this will be a list of lists, each is the list of observations in a hand
        self.obs_flat = list(chain(*self.observations))
        
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
            mask[2] = 1 # cannot fold if not facing a bet
        
        linspace = np.geomspace(.5, 2, num  = linspace_dim)
        stack_checker = lambda x: 1 if x * pot >= player_stack else 0
        mask[4:] += np.fromiter((stack_checker(x) for x in linspace), linspace.dtype) # mask away bets that are larger than stack or all in
        tensor_mask = torch.Tensor(mask)
        # apply mask
        curr_logits.masked_fill_(tensor_mask == 1, float('-inf'))

        # grad skip softmax -- neural replicator dynamics
        policy = self.softmax(curr_logits)

        np_dist = np.squeeze(policy[-1].detach().numpy())
        

        # SAMPLE
        action_index = np.random.choice(self.n_actions, p=np_dist)
        # calculate action log prob for use in advantage later
        alp = torch.log(policy[-1])[action_index] 

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

        return alp, action, policy
    
    def init_hands(self):
        # get all hands
        # run encoder for each of players
        for player in range(self.n_players):
            hand = self.env.get_hand(player)
            self.agent.init_player(player, hand)
    
    def chop_seq(self):
        #if length of observations is above a certain size, chop it back down to under sequence length by removing oldest hand
        #return flattened version to give to model on next run
        if len(self.observations) > self.max_sequence:
            self.observations = self.observations[1:]
            self.obs_flat = list(chain(*self.observations))

            self.rewards = self.rewards[1:]
            self.rewards_flat = list(chain(*self.rewards))
            

            self.values = self.values[1:]
            self.val_flat = list(chain(*self.values))

            self.action_log_probabilies = self.action_log_probabilies[1:]
            self.alp_flat = list(chain(*self.action_log_probabilies))

        else:

            self.obs_flat = list(chain(*self.observations))
            self.rewards_flat = list(chain(*self.rewards))
            self.val_flat = list(chain(*self.values))
            self.alp_flat = list(chain(*self.action_log_probabilies))

    def play_hand(self):
        # makes agent play one hand
        # deal cards
        rewards, observations = self.env.new_hand() # start a new hand
        player = self.env.in_turn
        self.init_hands() # pre load all of the hands

        # init lists for this hand
        self.observations.append(observations)
        self.rewards.append(rewards)
        self.chop_seq() # prepare for input to model
        self.values.append([])
        self.action_log_probabilies.append([])
        for x in range(len(rewards)):
                new_values = [-10000] * self.n_players #-10000 is filler value
                self.values[-1].append(torch.Tensor(new_values))
                new_alps = [-10000] * self.n_players
                self.action_log_probabilies[-1].append(torch.Tensor(new_alps))
        
        hand_over = False
        while not hand_over:
            # get values and policy -- should be in list form over sequence length
            policy_logits, values = self.agent(player, self.obs_flat)
            value = values.squeeze()[-1] # get last value estimate
            curr_logits = policy_logits[-1] # get last policy distribution

            #print('in_loop', values.requires_grad, curr_logits.requires_grad)

            alp, action, policy = self.sample_action(curr_logits) # handles mask, softmax, sample, detokenization
            rewards, obs, hand_over = self.env.take_action(action) # need to change environment to return hand_over boolean

            # add new information from this step
            self.rewards[-1] += rewards #add tensor
            self.observations[-1] += obs

            # value needs to be on a per player basis

            new_values = [-10000] * self.n_players #-10000 is filler value
            new_values[player] = value
            self.values[-1].append(torch.Tensor(new_values))
            for x in range(len(rewards) - 1):
                new_values = [-10000] * self.n_players #-10000 is filler value
                self.values[-1].append(torch.Tensor(new_values))

            new_alp = [-10000] * self.n_players
            new_alp[player] = alp
            self.action_log_probabilies[-1].append(torch.Tensor(new_alp))
            for x in range(len(rewards) - 1):
                new_alps = [-10000] * self.n_players
                self.action_log_probabilies[-1].append(torch.Tensor(new_alps))
            
            # prepare for next action
            self.chop_seq()

        
        Vals_T = [0] * self.n_players
        for player in range(self.n_players):
            _, V_T = self.agent(player, self.obs_flat)
            Vals_T[player] = V_T.squeeze()[-1]
        
        # process gradients and return loss:
        return self.get_loss(torch.stack(Vals_T))

    def get_loss(self, Vals_T):
        Qs = [0] * len(self.rewards_flat)
        Q_t = Vals_T
        for t in reversed(range(len(self.rewards_flat))):
            Q_t = self.rewards_flat[t] + self.gamma * Q_t #adds rewards up going backwards to get vals
            Qs[t] = Q_t
        
        Qs = torch.stack(Qs) #2d tensor sequence, players

        values = torch.stack(self.val_flat)

        # set Qs to filler value where value is filler value
        Qs.masked_fill(values == -10000, -10000)
        alps = torch.stack(self.alp_flat) 
        advantages = Qs - values 
        
        actor_loss = (-alps * advantages).mean() # loss function for policy going into softmax on backpass
        critic_loss = 0.5 * advantages.pow(2).mean() # autogressive critic loss - MSE
        
        loss = actor_loss + critic_loss
        print(critic_loss, critic_loss.requires_grad)
        print(actor_loss, actor_loss.requires_grad)
        print(loss.requires_grad)
        return loss
    

    def clear_memory(self):
        # zero the gradients of the model
        self.agent.model.zero_grad()
    