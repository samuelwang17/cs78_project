import torch
import torch.nn as nn

class Tokenizer():

    def __init__(self, token_dim_dict) -> None:
        # pos, action, pot, stack
        # card
        assert token_dim_dict['pot_edim'] + token_dim_dict['action_edim'] + token_dim_dict['pos_edim'] + token_dim_dict['stack_edim'] == token_dim_dict['card_edim']
        
        # embeddings that take value as argument:
        self.bet_embedding = nn.Sequential(nn.Linear(1, token_dim_dict['action_edim']), nn.ReLU(), nn.Linear(token_dim_dict['action_edim'], token_dim_dict['action_edim']), nn.ReLU())

        self.pot_embedding = nn.Sequential(nn.Linear(1, token_dim_dict['pot_edim']), nn.ReLU(), nn.Linear(token_dim_dict['pot_edim'], token_dim_dict['pot_edim']), nn.ReLU())

        self.stack_emdedding = nn.Sequential(nn.Linear(1, token_dim_dict['stack_edim']), nn.ReLU(), nn.Linear(token_dim_dict['stack_edim'], token_dim_dict['stack_edim']), nn.ReLU())

        # embeddings that require no arguments
        self.card_embedding = nn.Embedding(num_embeddings = 53, embedding_dim = token_dim_dict['card_edim'])
        self.player_embedding = nn.Embedding(num_embeddings = token_dim_dict['n_players'], embedding_dim = token_dim_dict['pos_edim']) # done by blinds ordering
        self.action_embedding = nn.Embedding(num_embeddings= 3, embedding_dim = token_dim_dict['action_edim']) # call, fold, win
    
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
        
            return self.card_embedding(torch.tensor(card_idx))

        else:
            # value dependent embeddings, player specific

            table_pos_embedded_tens = self.player_embedding(torch.tensor(observation['player']))
            player_stack_embedded_tens = self.stack_emdedding(torch.tensor(float(observation['stack'])).unsqueeze(-1))
            pot_embedded_tens = self.pot_embedding(torch.tensor(float(observation['pot'])).unsqueeze(-1))
            
            # win
            if observation['type'] == 'win':
                action_embedded_tens = self.action_embedding(torch.tensor(0))
            # call
            elif observation['type'] == 'call':
                action_embedded_tens = self.action_embedding(torch.tensor(1))

            # fold
            elif observation['type'] == 'fold':
                action_embedded_tens = self.action_embedding(torch.tensor(2))

            # bet/check
            else:
                action_embedded_tens = self.bet_embedding(torch.tensor(float(observation['value'])).unsqueeze(-1))
            
            # concat embeddings and return
            return torch.cat([table_pos_embedded_tens, action_embedded_tens, pot_embedded_tens, player_stack_embedded_tens]) # player, bet size, pot size
    

    def tokenize_list(self, observations):
        # convert list of observations to 2d tensor
        # each observation is a tensor
        seq = []
        for obs in observations:
            seq.append(self.tokenize(obs))
        
        obs_tensor = torch.stack(seq) #sequence, model_dim
        return obs_tensor
    
    def get_tokens(self, observations):
        obs_tensor = self.tokenize_list(observations)
        return obs_tensor
    