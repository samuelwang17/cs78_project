import torch

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
        suit = observation['suit']
        if suit == 'hearts':
            vec[11] = 1
        elif suit == 'diamonds':
            vec[12] = 1
        elif suit == 'spades':
            vec[13] = 1
        elif suit == 'clubs':
            vec[14] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']
        vec[30] = observation['p1']
        vec[31] = observation['p2']
        vec[32] = observation['p3']
        vec[33] = observation['p4']
        vec[34] = observation['p5']
        vec[35] = observation['p5']

    elif observation['type'] == 'bet':
        vec[7] = 1  # observation is type bet
        vec[observation['player']] = 1
        vec[28] = observation['value']

        # pot size and stack sizes
        vec[29] = observation['pot']
        vec[30] = observation['p1']
        vec[31] = observation['p2']
        vec[32] = observation['p3']
        vec[33] = observation['p4']
        vec[34] = observation['p5']
        vec[35] = observation['p5']


    elif observation['type'] == 'call':
        vec[8] = 1  # observation is type call
        vec[observation['player']] = 1
        vec[28] = observation['value']

        # pot size and stack sizes
        vec[29] = observation['pot']
        vec[30] = observation['p1']
        vec[31] = observation['p2']
        vec[32] = observation['p3']
        vec[33] = observation['p4']
        vec[34] = observation['p5']
        vec[35] = observation['p5']

    elif observation['type'] == 'fold':
        vec[9] = 1  # observation is type fold
        vec[observation['player']] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']
        vec[30] = observation['p1']
        vec[31] = observation['p2']
        vec[32] = observation['p3']
        vec[33] = observation['p4']
        vec[34] = observation['p5']
        vec[35] = observation['p5']

    elif observation['type'] == 'win':
        vec[10] = 1  # observation is type win
        vec[observation['player']] = 1

        # pot size and stack sizes
        vec[29] = observation['pot']
        vec[30] = observation['p1']
        vec[31] = observation['p2']
        vec[32] = observation['p3']
        vec[33] = observation['p4']
        vec[34] = observation['p5']
        vec[35] = observation['p5']

    return vec