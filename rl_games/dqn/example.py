from rl_games.dqn.nac_player import Player
from rl_games.core.play import play, play_many, play_human
from rl_games.games.nac import Nac

if __name__ == '__main__':
    g = Nac()
    p1 = Player('X')
    p2 = Player('O')
    play_many(g, [p1, p2], 1000)

    play_human(g, [p1, 'you'])
