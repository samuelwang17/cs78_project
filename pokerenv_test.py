from pokerenv import poker_env as poker


player_count = 6
game = poker(player_count)
print(game.new_hand())
exit()
while True:
    for i in range(player_count):
        print(game.get_hand(i))

    for i in range(player_count):
        action = {
            'type': 'bet',
            'player': i,
            'value': 2,
        }
        print(game.take_action(action))

    for i in range(player_count):
        action = {
            'type': 'bet',
            'player': i,
            'value': 2,
        }
        print(game.take_action(action))

    for i in range(player_count):
        action = {
            'type': 'bet',
            'player': i,
            'value': 2,
        }
        print(game.take_action(action))
    print(game.community_cards)
    for i in range(player_count):
        action = {
            'type': 'call',
            'player': i,
            'value': 2,
        }
        print(game.take_action(action))

