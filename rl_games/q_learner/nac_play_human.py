# Run with:
#     python -m rl_games.q_learner.nac_play_human

if __name__ == '__main__':
    from rl_games.q_learner.player import Player
    from rl_games.q_learner.play import play_human
    from rl_games.q_learner.nac import Nac
    from rl_games.q_learner.play import play_many

    game = Nac()
    x, o = Player('X'), Player('O')
    play_many(game, [x, o])
    play_many(game, [x, o])
    play_many(game, [x, o])
    play_many(game, [x, o])
    play_many(game, [x, o])
    winner = play_human(game, [x, 'you'])
    if winner:
        if isinstance(winner, str):
            print(f'Congratulations, the winner was {winner}!')
        else:
            print(f'The winner was {winner.id}')
    else:
        print('The game was a draw.')
