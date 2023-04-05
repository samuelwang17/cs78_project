import torch
import random

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

    def give_losses(self, out):
        loss, actor_loss, critic_loss, _ = out
        self.loss = loss
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss

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
        self.took_action = [
                               False] * self.n_players  # tracks whether players have taken action in a specific round of betting
        self.pot = 0
        self.stage = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river
        self.deck_position = 0

        # deal cards, pass to agents
        random.shuffle(self.deck)
        for i in range(self.n_players):
            self.hands += [self.get_next_cards(2)]

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

            if self.hand_count % self.hand_until_log == 0:
                with open(self.filename, 'a') as file:
                    self.history.append(f'Loss: {self.loss}, Actor Loss: {self.actor_loss}, Critic Loss: {self.critic_loss}')
                    self.history.append("\n\nHand End\n")
                    self.history.append("--------------------------------------------------------------------------------\n")
                    file.writelines(self.history)

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
                    self.history.append("\nShowdown win, " + str(p) + " wins " + str(self.pot / len(winners)))

            if self.hand_count % self.hand_until_log == 0:
                with open(self.filename, 'a') as file:
                    self.history.append("\n\nHand End\n")
                    self.history.append(
                        "--------------------------------------------------------------------------------\n")
                    self.history.append(f'Loss: {self.loss}, Actor Loss: {self.actor_loss}, Critic Loss: {self.critic_loss}')
                    file.writelines(self.history)

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

    def determine_showdown_winners(self):
        scores = [0] * self.n_players
        for p in range(self.n_players):
            if not self.in_hand[p]:
                continue

            cards = self.community_cards + self.hands[p]

            rank_count = [0] * 13
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
                suit_ranks = [0] * 13
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