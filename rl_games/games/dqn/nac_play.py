# Run with:
#     python -m rl_games.games.dqn.nac_play

from typing import Sequence, Tuple

from rl_games.dqn.dqn_player import DqnPlayer
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.player import Player
from rl_games.core.game import Game
from rl_games.games.nac import Nac, NacState, NacAction
from ..tqdm import range_with_timer
from .nac import NacDqnSetup


def get_sample_game_and_trained_players(num_rounds: int = 500, initial_explore_chance: float = 0.25) -> Tuple[Game, Sequence[Player]]:
    game = Nac()

    players = [
        DqnPlayer[NacState, NacAction]('X', NacDqnSetup(), explore_chance=initial_explore_chance),
        DqnPlayer[NacState, NacAction]('O', NacDqnSetup(), explore_chance=initial_explore_chance),
    ]

    play_many(game, players, range_with_timer(num_rounds), reduce_explore_chance=True)
    return game, players


if __name__ == '__main__':
    game1, players1 = get_sample_game_and_trained_players()
    play_human_ui(game1, players1)
