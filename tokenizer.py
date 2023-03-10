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

    def __init__(self, model_dim) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(36, 2*model_dim), # tokenizer has 36 dimensional output
            nn.ReLU(), # allows feature superposition in embedding
            nn.LayerNorm(2*model_dim),
            nn.Linear(2*model_dim, model_dim)
        )

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