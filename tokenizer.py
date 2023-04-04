import torch
import torch.nn as nn

def tokenize(observation):
    '''
    0-5 correspond to each player
    6-10 correspond to observation type (6: card, 7: bet, 8: call, 9: fold, 10: win)
    11-14 correspond to each suit (11: hearts, 12: diamonds, 13: spades, 14 clubs)
    15-27 correspond to each rank (2 through Ace)
    28 corresponds to bet amount (0 if not applicable)
    29 corresponds to pot size
    30-35 corresponds to the stack size of each player
    '''
    vec = torch.zeros(36)
    if observation['type'] == 'card':
        vec[6] = 1  # observation is type card
        vec[observation['rank'] + 13] = 1  # rank of card
        suit = observation['suit']
        if suit == 'h':
            vec[11] = 1
        elif suit == 'd':
            vec[12] = 1
        elif suit == 's':
            vec[13] = 1
        elif suit == 'c':
            vec[14] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']

    elif observation['type'] == 'bet':
        vec[7] = 1  # observation is type bet
        vec[observation['player']] = 1
        vec[28] = observation['value']

        # pot size and stack sizes
        vec[29] = observation['pot']

    elif observation['type'] == 'call':
        vec[8] = 1  # observation is type call
        vec[observation['player']] = 1
        vec[28] = observation['value']

        # pot size and stack sizes
        vec[29] = observation['pot']

    elif observation['type'] == 'fold':
        vec[9] = 1  # observation is type fold
        vec[observation['player']] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']

    elif observation['type'] == 'win':
        vec[10] = 1  # observation is type win
        vec[observation['player']] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']

    return vec

class Tokenizer(nn.Module):

    def __init__(self, n_players, token_dim_dict) -> None:
        super().__init__()
        # pos, action, pot, stack
        # card
        assert token_dim_dict['pot_edim'] + token_dim_dict['action_edim'] + token_dim_dict['pos_edim'] + token_dim_dict['stack_edim'] == token_dim_dict['card_edim']
        
        # embeddings that take value as argument:
        self.bet_embedding = nn.Sequential(nn.Linear(1, token_dim_dict['action_edim']), nn.ReLU(), nn.Linear(token_dim_dict['action_edim'], token_dim_dict['action_edim']), nn.ReLU())

        self.pot_embedding = nn.Sequential(nn.Linear(1, token_dim_dict['pot_edim']), nn.ReLU(), nn.Linear(token_dim_dict['pot_edim'], token_dim_dict['pot_edim']), nn.ReLU())

        # embeddings that require no arguments
        self.card_embedding = nn.Embedding(num_embeddings = 53, embedding_dim = token_dim_dict['card_edim'])
        self.player_embedding = nn.Embedding(num_embeddings = n_players, embedding_dim = token_dim_dict['pos_edim']) # done by blinds ordering
        self.action_embedding = nn.Embedding(num_embeddings= 3, embedding_dim = token_dim_dict['action_edim']) # call, fold, win
        self.stack_emdedding = nn.Embedding(num_embeddings=401, embedding_dim= token_dim_dict['stack_edim']) # player's stack can be in range 0-400 inclusive
    
    def tokenize(self, observation):
        '''
        for actions:
        all that matters is money into the pot -- everyone else will have to match this amount of money. 
        
        only other option is fold - this should get own token
        
        other options are wins and cards. cards already handled
        wins should get own token and value embedding

        so 4 value embeddings - one for money into pot, one for winning money, one for stack size, one for pot size
        then, 


        '''
        
        if observation['type'] == 'card':
            # complete
            suit = observation['suit']
            if suit == 'h':
                suit_idx = 0
            elif suit == 'd':
                suit_idx = 1
            elif suit == 's':
                suit_idx = 2
            elif suit == 'c':
                suit_idx = 3

            card_idx = suit_idx * 4 + (observation['rank'] - 2) # rank of card (2 through ace (14))
        
            return self.card_embedding(card_idx)

        else:
            # value dependent embeddings, player specific
            table_pos_embedded_tens = self.player_embedding(observation['player'])
            player_stack_embedded_tens = self.stack_emdedding(observation['stack'])
            pot_embedded_tens = self.pot_embedding(observation['pot'])
            
            # win
            if observation['type'] == 'win':
                action_embedded_tens = self.action_embedding[0]
            # call
            elif observation['type'] == 'call':
                action_embedded_tens = self.action_embedding[1]

            # fold
            elif observation['type'] == 'fold':
                action_embedded_tens = self.action_embedding[2]

            # bet/check
            else:
                action_embedded_tens = self.bet_embedding(observation['value'])
            
            # concat embeddings and return
            return torch.cat([table_pos_embedded_tens, action_embedded_tens, pot_embedded_tens, player_stack_embedded_tens]) # player, bet size, pot size
    

    def tokenize_list(self, observations):
        # convert list of observations to 2d tensor
        # each observation is a tensor
        seq = []
        for obs in observations:
            seq.append(tokenize(obs))
        
        obs_tensor = torch.stack(seq) #sequence, model_dim
        return obs_tensor
    
    def forward(self, observations):
        obs_tensor = self.tokenize_list(observations)
        token = self.embedding(obs_tensor) # sequence, model_dim
        return token
    