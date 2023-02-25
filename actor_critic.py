# total actor critic package, encapsulating a single worker/env pair

import torch
import torch.nn as nn
from transformer import mod_transformer as RLformer
from tokenizer import Tokenizer
from pokerenv import poker_env
from itertools import chain

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
        hand_tensor = Tokenizer(hand) #hand needs to be card observations -- list of length two of tensors
        self.register_buffer(f'hand_{player}', tensor= hand_tensor)

    def forward(self, player, obs_flat):
        #takes flattened inputs in list form, not tokenized
        enc_input = self.get_buffer(f'hand_{player}')
        dec_input = Tokenizer(obs_flat)
        policy, value = self.model(enc_input, dec_input)

        return policy, value

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
            attn_head = heads,
            sequence_length = max_sequence,
            enc_layers = enc_layers,
            dec_layers = dec_layers,
            action_dim = n_actions,
        )

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

        self.detokenize = None #detokenizer HERE

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
            self.rewards_flat = list(chain(*self.rewards_flat))

            self.values = self.values[1:]
            self.val_flat = list(chain(*self.values))

            self.action_log_probabilies = self.action_log_probabilies[1:]
            self.alp_flat = list(chain(*self.action_log_probabilies))

        else:
            self.obs_flat = list(chain(*self.observations))
            self.rewards_flat = list(chain(*self.rewards_flat))
            self.val_flat = list(chain(*self.values))
            self.alp_flat = list(chain(*self.action_log_probabilies))

    def play_hand(self):
        # makes agent play one hand
        
        # deal cards
        rewards, observations = self.env.new_hand() # start a new hand
        self.init_hands() # pre load all of the hands

        # init lists for this hand
        self.observations += [observations] 
        self.rewards += [rewards]

        self.chop_seq() # prepare for input to model
        
        hand_over = False
        while not hand_over:                

            # get values and policy -- should be in list form over sequence length
            policy, values = self.agent(self.obs_flat)
            value = values[-1].detach().numpy()[0,0] # get last value estimate
            dist = policy[-1].detach().numpy() # get last policy distribution

            # randomly sample an action
            action = np.random.choice(self.n_actions, p=np.squeeze(dist))

            # UNFINISHED: Need to detokenize actions HERE
            action = self.detokenize(action)

            alp = torch.log(policy.squeeze(0)[action])
            reward, obs, hand_over = self.env.take_action(action) # need to change environment to return hand_over boolean

            # add new information from this step
            self.rewards[-1].append(reward)
            self.observations[-1].append(obs)
            self.values[-1].append(value)
            self.action_log_probabilies.append(alp)
            
            # prepare for next action
            self.chop_seq()
        
        V_T, _ = self.agent(self.obs_flat)
        
        # process gradients and return loss:
        return self.get_loss(V_T)

    def get_loss(self, values, rewards, V_T):

        Qs = []
        Q_t = V_T
        for t in reversed(range(len(rewards))):
            Q_t = rewards[t] + self.gamma * Q_t
            Qs[t] = Q_t
        
        Qs = torch.FloatTensor(Qs)
        values = torch.FloatTensor(self.val_flat)
        alps = torch.stack(self.alp_flat)
        advantages = Qs - values

        
        actor_loss = (-alps * advantages).mean() # loss function for policy going into softmax on backpass
        critic_loss = 0.5 * advantages.pow(2).mean() # autogressive critic loss - MSE
        loss = actor_loss + critic_loss # no entropy in this since that would deviate from deepnash
        return loss
    