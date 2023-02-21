class poker_env():
    '''
    Texas no-limit holdem environment.
    '''

    def __init__(self, n_players) -> None:
        
        self.n_players = n_players
        
        self.stacks = {}
        for player in range(n_players):
            self.stacks[player] = 200
        
        self.button = 0 # button starts at player 0 WLOG
        
        # self.hands -- TO BE IMPLEMENTED
    
    def take_action(self, action):
        '''
        Only function that is externally called in training
        Takes an action, returns a rewards tensor which has an element for each player, and a list of observations. 
        Observations are all public information -- does not include dealt hands
        Moves game state to next point where action input is required
        Rewards implementation currently changing -- very fucked up rn
        '''
        rewards = {} # CHANGE TO TENSOR
        for p in range(self.n_players):
            rewards[p] = 0
        
        observations = [action] # first observation returned is always the action being taken
        player = action['player']
        type = action['type'] # action type is one of {bet, call, fold}
        value = action['value']

        self.took_action[player] = True

        if type == 'bet':
            # move money from player to pot
            self.stacks[player] -= value
            self.pot += value
            self.current_bet += value # bets are valued independently and are NOT measured by cumulative sum -- current_bet tracks that

            # reward is negative of amount bet
            rewards[player] -= value

            # other players are now behind the bet
            for x in range(self.n_players):
                self.behind[x] += value
            
            # player who just bet cannot be behind
            self.behind[player] = 0

        if type == 'call':
            # need to catch up to current bet
            call_size = self.behind[player]
            self.behind[player] = 0

            # move money from player to pot
            self.stacks[player] -= call_size
            self.pot += call_size
            self.current_bet += call_size # bets are valued independently and are NOT measured by cumulative sum -- current_bet tracks that

            # reward is negative of amount bet
            rewards[player] = -1 * call_size
        
        if type == 'fold':
            # player becomes inactive
            self.in_hand[player] = False


        # if everyone is square or folded, advance to next game stage
        square_check = True
        for p in range(self.n_players):
            if (self.in_hand[p] and self.behind[p] != 0) or not self.took_action[x]: # Big blind option handled via took_action
                square_check = False
        
        if square_check:
            # advance stage, and any other subcalls that come with that
            advance_stage_rewards, advance_stage_observations = self.advance_stage()
            rewards += advance_stage_rewards
            observations += advance_stage_observations
        
        else:
            # advance to next player
            self.in_turn = (self.in_turn + 1) % self.n_players

        return rewards, observations
    
    def advance_stage(self):
        # this is called anytime that there is no player who is: 1. in the hand, 2. behind the bet, and 3. has not taken action

        # payout if only one player is left
        if sum(self.in_hand) == 1:
            for p in range(self.n_players):
                if self.in_hand[p]:
                    # payout!
                    advance_stage_rewards ## ISSUE -- REWARDS NEED TO BE TAGGED WITH PLAYER, CHANGED IN FORMAT TO TENSOR
                    # advance_stage_observations = PRINT OF PAYOUT
            new_hand_rewards, new_hand_observations  = self.new_hand() # move on to next hand
            advance_stage_rewards += new_hand_rewards
            advance_stage_observations += new_hand_observations

        # advance stage if not river
        elif self.stage != 3:
            self.stage += 1
            for p in range(self.n_players):
                if self.in_hand[p]: # this keeps took_action true for players who have folded to save a conditional above
                     self.took_action[p] = False
            advance_stage_rewards, advance_stage_observations = self.card_reveal()

        # compare hands and payout, then deal new hand
        else:
            # INSERT HAND COMPARISON HERE
            pass

        return advance_stage_rewards, advance_stage_observations
    
    def new_hand(self):
    
        self.button = (self.button + 1) % self.n_players
        self.in_turn = (self.button + 1) % self.n_players
        self.behind = [0] * self.n_players
        self.in_hand = [True] * self.n_players
        self.took_action = [False] * self.n_players # tracks whether players have taken action in a specific round of betting
        self.pot = 0
        self.current_bet = 0
        self.stage = 0 # 0: pre-flop, 1: flop, 2: turn, 3: river

        #big blind is 2, small blind is 1
        small_blind = {'player': self.in_turn, 'type': 'bet', 'value': 1}
        rewards_1, observations_1 = self.take_action(small_blind)

        big_blind_player = self.in_turn
        big_blind = {'player': big_blind_player, 'type': 'bet', 'value': 2}
        rewards_2, observations_2 =  self.take_action(big_blind)
        self.took_action[big_blind_player] = False

        rewards_1 += rewards_2
        observations_1 += observations_2

        # NEED TO DEAL CARDS HERE, PASS THEM TO AGENTS (use self.hands)
        
        return rewards_1, observations_1
    
    def card_reveal(self):
        
        if self.stage == 0:
            # revealing the flop
            card_rewards = [0, 0, 0] #card reveals have reward zero
            # card_observations = INSERT CARD DRAW FUNCTION HERE
        else:
            # one card to be revealed
            card_rewards = [0]
            # card_observations = INSERT CARD DRAW FUNCTION HERE
        self.in_turn = (self.button + 1) % self.n_players

        return card_rewards, card_observations