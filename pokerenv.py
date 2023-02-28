import random
import numpy as np


class poker_env():
    '''
    Texas no-limit holdem environment.
    '''

    def __init__(self, n_players, batch_size) -> None:
        self.n_players = n_players

        self.stacks = [[0 for i in range(n_players)] for j in range(batch_size)]

        self.button = 0  # button starts at player 0 WLOG

        self.deck = []
        for suit in ["h", "d", "s", "c"]:
            for rank in range(2, 15):
                self.deck += [[suit, rank]]

        self.rank_mapping = {}
        for i in range(2, 11):
            self.rank_mapping[str(i)] = str(i)
        self.rank_mapping['11'] = "J"
        self.rank_mapping['12'] = "Q"
        self.rank_mapping['13'] = "K"
        self.rank_mapping['14'] = "A"

        self.filename = "hand_replays.txt"
        self.hand_count = 0
        self.hand_until_log = 100

        self.batch_size = batch_size

    def new_hand(self):
        self.hand_count += 1
        self.history = []

        if self.hand_count % self.hand_until_log == 0:
            self.history.append(
                "\n\n--------------------------------------------------------------------------------\n")
            self.history.append("Hand " + str(self.hand_count) + " Start\n")

        for i in range(self.batch_size):
            for player in range(self.n_players):
                self.stacks[i][player] = 200

        self.community_cards = []
        self.hands = []
        self.deck_position = 0


        self.button = (self.button + 1) % self.n_players

        self.in_turn = [(self.button + 1) % self.n_players for i in range(self.batch_size)]

        self.behind = [[0 for i in range(self.n_players)] for j in range(self.batch_size)]
        self.current_bets = [[0 for i in range(self.n_players)] for j in range(self.batch_size)]
        self.in_hand = [[True for i in range(self.n_players)] for j in range(self.batch_size)]
        self.took_action = [[False for i in range(self.n_players)] for j in range(
            self.batch_size)]  # tracks whether players have taken action in a specific round of betting
        self.pot = [0 for i in range(self.batch_size)]
        self.stage = [0 for i in range(self.batch_size)]  # 0: pre-flop, 1: flop, 2: turn, 3: river
        self.hand_overs = [False for i in range(self.batch_size)]

        # deal cards, pass to agents
        random.shuffle(self.deck)
        for x in range(self.n_players):
            self.hands += [self.get_next_cards(2)]

        small_blinds = [{'player': self.in_turn[0], 'type': 'bet', 'value': 1, 'pot': 0}] * self.batch_size
        rewards, observations, hand_over = self.take_actions(small_blinds)

        big_blind_player = self.in_turn[0]
        big_blinds = [{'player': big_blind_player, 'type': 'bet', 'value': 2, 'pot': 1}] * self.batch_size
        r, o, hand_over = self.take_actions(big_blinds)

        for i in range(self.batch_size):
            self.took_action[i][big_blind_player] = False

        for i in range(self.batch_size):
            rewards[i].append(r[i])
            observations[i].append(o[i])

        return rewards, observations

    def get_hand(self, player):
        if len(self.hands) == 0:
            return None
        card_observations = []
        for card in self.hands[player]:
            card_observations += [
                {'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': 0}]

        return card_observations

    def take_actions(self, actions):
        '''
        Only function that is externally called in training
        Takes an action, returns a rewards tensor which has an element for each player, and a list of observations.
        Observations are all public information -- does not include dealt hands
        Moves game state to next point where action input is requires
        '''
        rewards_batch = []
        observations_batch = []
        hand_over_batch = []
        for i in range(self.batch_size):

            rewards = [[0 for i in range(self.n_players)]]
            if self.hand_overs[i]:
                rewards_batch.append(rewards)
                observations_batch.append([])
                hand_over_batch.append(True)
                continue

            hand_over = False
            player = actions[i]['player']
            type = actions[i]['type']  # action type is one of {bet, call, fold}
            value = actions[i]['value']

            self.took_action[i][player] = True

            if type == 'bet':
                # move money from player to pot
                self.stacks[i][player] -= value
                self.pot[i] += value
                # reward is negative of amount bet
                rewards[0][player] = -value

                # other players are now behind the bet
                for x in range(self.n_players):
                    self.behind[i][x] = max(self.behind[i][x],
                                            value + self.current_bets[i][player] - self.current_bets[i][x])

                self.current_bets[i][player] += value

                # player who just bet cannot be behind
                self.behind[i][player] = 0

            if type == 'call':
                # need to catch up to current bet
                call_size = self.behind[i][player]
                actions[i]['value'] = call_size
                self.current_bets[i][player] += call_size

                # move money from player to pot
                self.stacks[i][player] -= call_size
                self.pot[i] += call_size

                self.behind[i][player] = 0

                # reward is negative of amount bet
                rewards[0][player] = -1 * call_size

            if type == 'fold':
                # player becomes inactive
                self.in_hand[i][player] = False

            actions[i]['pot'] = self.pot[i]
            observations = [actions[i]]

            if self.hand_count % self.hand_until_log == 0:
                if i == 0:
                    self.history.append("\n\nPlayer's Cards: \n")
                    for i in range(2):
                        self.history.append(
                            self.rank_mapping[str(self.hands[player][i][1])] + str(self.hands[player][i][0]) + ", ")
                    self.history.append("\nCommunity Cards: \n")
                    for i in range(len(self.community_cards)):
                        self.history.append(
                            self.rank_mapping[str(self.community_cards[i][1])] + str(self.community_cards[i][0]) + ", ")
                    if len(self.community_cards) == 0:
                        self.history.append("Preflop")
                    self.history.append("\nAction: \n")
                    self.history.append(str(actions[i]))

            # if everyone is square or folded, advance to next game stage
            square_check = True
            for p in range(self.n_players):
                if (self.in_hand[i][p] and self.behind[i][p] != 0) or not self.took_action[i][
                    p]:  # Big blind option handled via took_action
                    square_check = False

            hand_over = False
            if square_check or sum(self.in_hand[i]) == 1:
                # advance stage, and any other subcalls that come with that
                advance_stage_rewards, advance_stage_observations, hand_ovr = self.advance_stage(i)
                if hand_ovr:
                    hand_over = True
                rewards += advance_stage_rewards
                observations += advance_stage_observations

            else:
                # advance to next player
                self.in_turn[i] = (self.in_turn[i] + 1) % self.n_players

            self.hand_overs[i] = hand_over
            rewards_batch.append(rewards)
            observations_batch.append(observations)
            hand_over_batch.append(False)
        return rewards_batch, observations_batch, hand_over_batch

    def advance_stage(self, index):
        # this is called anytime that there is no player who is: 1. in the hand, 2. behind the bet, and 3. has not taken action
        advance_stage_rewards = [[0 for i in range(self.n_players)]]
        advance_stage_observations = []
        hand_over = False

        for x in range(self.n_players):
            self.behind[index][x] = 0
            self.current_bets[index][x] = 0

        # payout if only one player is left
        if sum(self.in_hand[index]) == 1:
            for p in range(self.n_players):
                if self.in_hand[index][p]:
                    # payout!
                    advance_stage_rewards[0][p] += self.pot[index]
                    self.stacks[index][p] += self.pot[index]
                    advance_stage_observations += [
                        {'player': p, 'type': 'win', 'value': self.pot[index], 'pot': self.pot[index]}]
                    if self.hand_count % self.hand_until_log == 0 and index == 0:
                        self.history.append("\nFolds around, player " + str(p) + " wins " + str(self.pot[index]))

            if self.hand_count % self.hand_until_log == 0 and index == 0:
                with open(self.filename, 'a') as file:
                    self.history.append("\n\nHand End\n")
                    self.history.append(
                        "--------------------------------------------------------------------------------\n")
                    file.writelines(self.history)

            hand_over = True

        # advance stage if not river
        elif self.stage[index] != 3:
            self.stage[index] += 1
            for p in range(self.n_players):
                if self.in_hand[index][
                    p]:  # this keeps took_action true for players who have folded to save a conditional above
                    self.took_action[index][p] = False
            advance_stage_rewards, advance_stage_observations = self.card_reveal(index)

        # compare hands and payout, then deal new hand
        else:
            winners = self.determine_showdown_winners(index)
            for p in winners:
                advance_stage_rewards[0][p] += self.pot[index] / len(winners)
                self.stacks[index][p] += self.pot[index] / len(winners)
                advance_stage_observations += [{'player': p, 'type': 'win', 'value': self.pot[index] / len(winners),
                                                'pot': self.pot[index]}]
                if self.hand_count % self.hand_until_log == 0 and index == 0:
                    self.history.append("\nShowdown win, " + str(p) + " wins " + str(self.pot / len(winners)))

            if self.hand_count % self.hand_until_log == 0 and index == 0:
                with open(self.filename, 'a') as file:
                    self.history.append("\n\nHand End\n")
                    self.history.append(
                        "--------------------------------------------------------------------------------\n")
                    file.writelines(self.history)

            hand_over = True

        return advance_stage_rewards, advance_stage_observations, hand_over

    def card_reveal(self, index):

        if self.stage == 1:
            # revealing the flop
            card_rewards = [[0] * self.n_players] * 3  # card reveals have reward zero
            if len(self.community_cards) == 3:
                cards = self.community_cards[0:3]
            else:
                cards = self.get_next_cards(3)
                self.community_cards += cards
            card_observations = []
            for card in cards:
                card_observations += [{'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': self.pot[index]}]
        else:
            card_rewards = [[0] * self.n_players]
            if len(self.community_cards) == self.stage[index] + 2:
                card = self.community_cards[self.stage[index] + 1]
            else:
                # one card to be revealed
                card = self.get_next_cards(1)
                self.community_cards += [card]
            card_observations = [
                {'type': 'card', 'suit': card[0], 'rank': card[1], 'pot': self.pot[index]}]

        self.in_turn[index] = (self.button + 1) % self.n_players
        return card_rewards, card_observations

    def get_next_cards(self, num_cards):
        if num_cards == 1:
            card = self.deck[self.deck_position]
            self.deck_position += 1
            return card
        elif num_cards == 2:
            cards = [self.deck[self.deck_position], self.deck[self.deck_position + 1]]
            self.deck_position += 2
            return cards
        elif num_cards == 3:
            cards = [self.deck[self.deck_position], self.deck[self.deck_position + 1],
                     self.deck[self.deck_position + 2]]
            self.deck_position += 3
            return cards
        return None

    def determine_showdown_winners(self, index):
        scores = [0 for i in range(self.n_players)]
        for p in range(self.n_players):
            if not self.in_hand[index][p]:
                continue

            cards = self.community_cards + self.hands[p]

            rank_count = [0 for i in range(13)]
            suit_count = {"h": 0, "d": 0, "s": 0, "c": 0}

            for card in cards:
                suit_count[card[0]] += 1
                rank_count[card[1] - 2] += 1

            # find rank with highest count and rank with second highest count
            first_count = 0
            second_count = 0
            first_rank = 0
            second_rank = 0
            straight_count = 0
            straight_high = 0
            for rank in range(2, 15):
                current_count = rank_count[rank - 2]
                if current_count > first_count:
                    second_count = first_count
                    second_rank = first_rank
                    first_count = current_count
                    first_rank = rank
                elif current_count == first_count:
                    if rank > first_rank:
                        second_count = first_count
                        second_rank = first_rank
                        first_count = current_count
                        first_rank = rank
                    elif current_count == second_count:
                        second_rank = rank
                    else:
                        second_count = current_count
                        second_rank = rank
                elif current_count > second_count:
                    second_count = rank
                    second_rank = rank
                elif current_count == second_count:
                    second_rank = rank

                if current_count == 0:
                    continue

                if rank == 2:
                    if rank_count[12] > 0:
                        straight_count = 2
                else:
                    if rank_count[rank - 3] > 0:
                        straight_count += 1
                    else:
                        straight_count = 1

                    if straight_count >= 5:
                        straight_high = rank

            # check for flush
            flush_high = 0
            flush_suit = ""
            for suit in suit_count:
                if suit_count[suit] >= 5:
                    flush_suit = suit
                    for card in cards:
                        if card[0] == suit:
                            flush_high = max(flush_high, card[1])

            # check for straight flush
            if flush_high != 0 and straight_high != 0:
                sf_high = 0
                sf_count = 0
                suit_ranks = [0 for i in range(13)]
                for card in cards:
                    if card[0] == flush_suit:
                        suit_ranks[card[1] - 2] = 1

                for rank in range(2, 15):
                    if rank == 2:
                        if suit_ranks[12]:
                            sf_count = 2
                        else:
                            if suit_ranks[rank - 3]:
                                sf_count += 1
                            else:
                                sf_count = 1

                            if sf_count >= 5:
                                sf_high = rank

                if sf_high != 0:
                    scores[p] = 27 + (0.2 * sf_high)
                    continue

            # quads
            if first_count == 4:
                scores[p] = 24 + (0.2 * first_rank)
                continue

            # full house
            if first_count == 3 and second_count >= 2:
                scores[p] = 21 + 0.2 * first_rank + 0.01 * second_rank
                continue

            # flush
            if flush_high != 0:
                scores[p] = 18 + (0.2 * flush_high)
                continue

            # straight
            if straight_high != 0:
                scores[p] = 15 + (0.2 * straight_high)
                continue

            # sort ranks now that high cards matter
            ranks = []
            for card in cards:
                ranks.append(card[1])
            ranks.sort()

            # trips
            if first_count == 3:
                high = 0
                second_high = 0
                pos = 6
                while high == 0 and second_high == 0:
                    if ranks[pos] == first_rank:
                        pass
                    elif high == 0:
                        high = ranks[pos]
                    else:
                        second_high = ranks[pos]
                    pos -= 1

                scores[p] = 12 + 0.2 * high + 0.01 * second_high
                continue

            # two pair
            if first_count == 2 and second_count == 2:
                high = 0
                pos = 6
                while high == 0:
                    if ranks[pos] == first_rank or ranks[pos] == second_rank:
                        pass
                    else:
                        high = ranks[pos]

                    pos -= 1

                scores[p] = 9 + 0.2 * first_rank + 0.01 * second_rank + 0.002 * high
                continue

            # pair
            if first_count == 2:
                high = 0
                second_high = 0
                third_high = 0
                pos = 6
                while high == 0 and second_high == 0 and third_high == 0:
                    if ranks[pos] == first_rank or ranks[pos] == second_rank:
                        pass
                    elif high == 0:
                        high = ranks[pos]
                    elif second_high == 0:
                        second_high = ranks[pos]
                    else:
                        third_high = ranks[pos]
                    pos -= 1
                scores[p] = 6 + 0.2 * first_rank + 0.01 * high + 0.002 * second_high + 0.0005 * third_high
                continue

            # high card
            scores[p] = 0.2 * ranks[6] + 0.01 * ranks[5] + 0.002 * ranks[4] + 0.0005 * ranks[3] + 0.00001 * ranks[2]
            continue

        max_score = 0
        winners = []
        for p in range(len(scores)):
            if scores[p] > max_score:
                winners = [p]
                max_score = scores[p]
            elif scores[p] == max_score:
                winners += [p]

        return winners
