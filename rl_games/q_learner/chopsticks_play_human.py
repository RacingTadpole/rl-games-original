# Run with:
#     python -m rl_games.q_learner.chopsticks_play_human

if __name__ == '__main__':
    from rl_games.q_learner.player import Player
    from rl_games.q_learner.play import play_human
    from rl_games.q_learner.chopsticks import Chopsticks
    from rl_games.q_learner.play import play_many

    players = [Player('P1'), Player('P2')]
    game = Chopsticks()

    play_many(game, players, 25000)
    for player in players:
        player.explore_chance = 0.1
    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0.05
    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0

    try:
        index = int(input(f'Play as player number {tuple(range(1, 1 + len(players)))}? '))
    except ValueError:
        index = 1
    players[index - 1] = 'human'
    winner = play_human(game, players)

    if winner:
        if isinstance(winner, str):
            print(f'Congratulations, the winner was {winner}!')
        else:
            print(f'The winner was {winner.id}')
    else:
        print('The game was a draw.')
