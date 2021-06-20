# Run with:
#     python -m rl_games.games.dqn.nac.profile

from typing import Sequence, Tuple
import cProfile
import pstats
import io
from pstats import SortKey

from rl_games.dqn.dqn_player import DqnPlayer
from rl_games.core.play import play_many
from rl_games.core.player import Player
from rl_games.core.game import Game
from rl_games.games.nac import Nac, NacState, NacAction
from rl_games.games.dqn.nac.setup import NacDqnSetup


def get_sample_game_and_trained_players(num_rounds: int = 50, initial_explore_chance: float = 0.25) -> Tuple[Game, Sequence[Player]]:
    game = Nac()

    players = [
        DqnPlayer[NacState, NacAction](game.markers[0], NacDqnSetup(), explore_chance=initial_explore_chance),
        DqnPlayer[NacState, NacAction](game.markers[1], NacDqnSetup(), explore_chance=initial_explore_chance),
    ]

    play_many(game, players, range(num_rounds), reduce_explore_chance=True)
    return game, players


if __name__ == '__main__':
    with cProfile.Profile() as profile:
        get_sample_game_and_trained_players()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
