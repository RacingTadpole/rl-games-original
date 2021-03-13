# Run with:
#     python -m rl_games.q_learner.nac_play_human

if __name__ == '__main__':
    from rl_games.q_learner.player import Player
    from rl_games.q_learner.play import play_human
    from rl_games.q_learner.nac import Nac
    from rl_games.q_learner.play import play_many

    game = Nac()
    players = Player('X'), Player('O')

    play_many(game, players, 2000)
    for player in players:
        player.explore_chance = 0.1
    play_many(game, players, 2000)
    for player in players:
        player.explore_chance = 0.05
    play_many(game, players, 2000)
    for player in players:
        player.explore_chance = 0

    winner = play_human(game, [players[0], 'human'])
    if winner:
        if isinstance(winner, str):
            print(f'Congratulations, the winner was {winner}!')
        else:
            print(f'The winner was {winner.id}')
    else:
        print('The game was a draw.')
