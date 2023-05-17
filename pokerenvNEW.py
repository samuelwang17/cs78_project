import torch
import random
import numpy as np
import eval7 as e7

class poker_env():
    '''
    Texas no-limit holdem environment.
    '''

    def __init__(self, n_players) -> None:
        self.action_count = 0

        self.n_players = n_players

        self.stacks = [0] * n_players

        self.button = 0  # button starts at player 0 WLOG

        self.current_largest_bet = 1
        
        deck = e7.Deck()
        self.deck = np.array(deck.deal(52))

        self.filename = "hand_replays.txt"
        self.hand_count = 0
        self.hand_until_log = 50

    def end_hand(self, out, adv):
        loss, actor_loss, critic_loss, _ = out
        if self.hand_count % self.hand_until_log == 0:
            with open(self.filename, 'a') as file:
                self.history.append(f'Loss: {loss}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}')
                self.history.append(f"\n {adv} \n")
                self.history.append("\n\nHand End\n")
                self.history.append("--------------------------------------------------------------------------------\n")
                file.writelines(self.history)

    def new_hand(self):
        self.hand_count += 1
        self.history = []

        if self.hand_count % self.hand_until_log == 0:
            self.history.append("\n\n--------------------------------------------------------------------------------\n")
            self.history.append("Hand " + str(self.hand_count) + " Start\n")
        
        for player in range(self.n_players):
            self.stacks[player] = 200
        self.community_cards = []
        self.hands = []
        self.deck_position = 0
        self.button = (self.button + 1) % self.n_players
        self.in_turn = (self.button + 1) % self.n_players
        self.behind = [0] * self.n_players
        self.current_bets = [0] * self.n_players
        self.current_largest_bet = 1
        self.in_hand = [True] * self.n_players
        self.took_action = [False] * self.n_players  # tracks whether players have taken action in a specific round of betting
        self.pot = 0
        self.stage = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river
        self.deck_position = 0

        # deal cards, pass to agents
        indices = np.random.randint(52, size=9) #will need to change beyong 2 players
        self.board = self.deck[indices[4:]]

        self.hands = [self.deck[indices[:2]], self.deck[indices[2:4]]]
        

        # big blind is 2, small blind is 1
        small_blind = {'player': self.in_turn, 'type': 'bet', 'value': 1, 'pot': self.pot}
        rewards_1, observations_1, hand_over = self.take_action(small_blind)

        big_blind_player = self.in_turn
        big_blind = {'player': big_blind_player, 'type': 'bet', 'value': 2, 'pot': self.pot}
        rewards_2, observations_2, hand_over = self.take_action(big_blind)
        self.took_action[big_blind_player] = False

        rewards_1 += rewards_2
        observations_1 += observations_2

        return rewards_1, observations_1

    def get_hand(self, player):
        if len(self.hands) == 0:
            return None
        card_observations = []
        for card in self.hands[player]:
            card_observations += [
                {'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': self.pot}]

        return card_observations

    def take_action(self, action):
        '''
        Only function that is externally called in training
        Takes an action, returns a rewards tensor which has an element for each player, and a list of observations.
        Observations are all public information -- does not include dealt hands
        Moves game state to next point where action input is required
        Rewards implementation currently changing -- very fucked up rn
        '''
        self.action_count += 1

        rewards = [torch.zeros(self.n_players)]

        player = action['player']
        type = action['type']  # action type is one of {bet, call, fold}
        value = action['value']

        self.took_action[player] = True

        if type == 'bet':
            # move money from player to pot
            self.stacks[player] -= value
            self.pot += value
            # reward is negative of amount bet
            rewards[0][player] = -value

            # other players are now behind the bet
            for x in range(self.n_players):
                self.behind[x] = max(self.behind[x], value + self.current_bets[player] - self.current_bets[x])

            self.current_bets[player] += value

            self.current_largest_bet = self.current_bets[player]

            # player who just bet cannot be behind
            self.behind[player] = 0

        if type == 'call':
            # need to catch up to current bet
            call_size = self.behind[player]
            action['value'] = call_size
            self.current_bets[player] += call_size

            # move money from player to pot
            self.stacks[player] -= call_size
            self.pot += call_size

            self.behind[player] = 0

            # reward is negative of amount bet
            rewards[0][player] = -1 * call_size

        if type == 'fold':
            # player becomes inactive
            self.in_hand[player] = False


        action['pot'] = self.pot
        action['stack'] = self.stacks[player]
        observations = [action]

        if self.hand_count % self.hand_until_log == 0:
            self.history.append("\n\nPlayer's Cards: \n")
            for i in range(2):
                self.history.append(self.rank_mapping[str(self.hands[player][i][1])] + str(self.hands[player][i][0]) + ", ")
            self.history.append("\nCommunity Cards: \n")
            for i in range(len(self.community_cards)):
                self.history.append(self.rank_mapping[str(self.community_cards[i][1])] + str(self.community_cards[i][0]) + ", ")
            if len(self.community_cards) == 0:
                self.history.append("Preflop")
            self.history.append("\nAction: \n")
            self.history.append(str(action))

        # if everyone is square or folded, advance to next game stage
        square_check = True
        for p in range(self.n_players):
            if (self.in_hand[p] and self.behind[p] != 0) or not self.took_action[
                p]:  # Big blind option handled via took_action
                square_check = False

        hand_over = False
        if square_check or sum(self.in_hand) == 1:
            # advance stage, and any other subcalls that come with that
            advance_stage_rewards, advance_stage_observations, hand_ovr = self.advance_stage()
            if hand_ovr:
                hand_over = True
            rewards += advance_stage_rewards
            observations += advance_stage_observations

        else:
            # advance to next player
            self.in_turn = (self.in_turn + 1) % self.n_players

        return rewards, observations, hand_over

    def advance_stage(self):
        # this is called anytime that there is no player who is: 1. in the hand, 2. behind the bet, and 3. has not taken action
        advance_stage_rewards = [torch.zeros(self.n_players)]
        advance_stage_observations = []
        hand_over = False

        for x in range(self.n_players):
            self.behind[x] = 0
            self.current_bets[x] = 0
        self.current_largest_bet = 0

        # payout if only one player is left
        if sum(self.in_hand) == 1:
            for p in range(self.n_players):
                if self.in_hand[p]:
                    # payout!
                    advance_stage_rewards[0][p] += self.pot
                    self.stacks[p] += self.pot
                    advance_stage_observations += [{'player': p, 'type': 'win', 'value': self.pot, 'pot': self.pot, 'stack': self.stacks[p]}]
                    if self.hand_count % self.hand_until_log == 0:
                        self.history.append("\nFolds around, player " + str(p) + " wins " + str(self.pot)+ "\n")

            hand_over = True

        # advance stage if not river
        elif self.stage != 3:
            self.stage += 1
            for p in range(self.n_players):
                if self.in_hand[
                    p]:  # this keeps took_action true for players who have folded to save a conditional above
                    self.took_action[p] = False
            advance_stage_rewards, advance_stage_observations = self.card_reveal()

        # compare hands and payout, then deal new hand
        else:
            winners = self.determine_showdown_winners()
            for p in winners:
                advance_stage_rewards[0][p] += self.pot / len(winners)
                self.stacks[p] += self.pot / len(winners)
                advance_stage_observations += [{'player': p, 'type': 'win', 'value': self.pot / len(winners),
                                               'pot': self.pot, 'stack': self.stacks[p]}]
                if self.hand_count % self.hand_until_log == 0:
                    self.history.append("\nShowdown win, " + str(p) + " wins " + str(self.pot / len(winners)) + '\n')

            hand_over = True

        return advance_stage_rewards, advance_stage_observations, hand_over

    def card_reveal(self):

        if self.stage == 1:
            # revealing the flop
            card_rewards = [torch.zeros(self.n_players)] * 3  # card reveals have reward zero
            cards = self.get_next_cards(3)
            self.community_cards += cards
            card_observations = []
            for card in cards:
                card_observations += [{'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': self.pot}]
        else:
            # one card to be revealed
            card_rewards = torch.zeros(self.n_players)
            card = self.get_next_cards(1)
            self.community_cards += [card]
            card_observations = [
                {'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': self.pot}]
        self.in_turn = (self.button + 1) % self.n_players

        return card_rewards, card_observations

    def get_next_cards(self, num_cards):
        if num_cards == 1:
            card = self.board[self.deck_position]
            self.deck_position += 1
            return card
        elif num_cards == 2:
            cards = [self.board[self.deck_position], self.board[self.deck_position + 1]]
            self.deck_position += 2
            return cards
        elif num_cards == 3:
            cards = [self.board[self.deck_position], self.board[self.deck_position + 1],
                     self.board[self.deck_position + 2]]
            self.deck_position += 3
            return cards
        return None
        


    def determine_showdown_winners(self):
        
        eval0 = e7.evaluate(self.hands[0])
        eval1 = e7.evaluate(self.hands[0])

        if eval0 > eval1:
            return [0]
        elif eval1 > eval0:
            return[1]
        else:
            return [0,1]