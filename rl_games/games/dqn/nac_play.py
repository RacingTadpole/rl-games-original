# Run with:
#     python -m rl_games.games.dqn.nac_play

from tqdm import tqdm
from typing import Sequence, Tuple

from rl_games.dqn.dqn_player import DqnPlayer
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.player import Player
from rl_games.core.game import Game

from rl_games.games.nac import Nac, NacState, NacAction
from .nac import NacDqnSetup


def get_sample_game_and_trained_players() -> Tuple[Game, Sequence[Player]]:
    game = Nac()

    players = [
        DqnPlayer[NacState, NacAction]('X', NacDqnSetup(), explore_chance=0.25),
        DqnPlayer[NacState, NacAction]('O', NacDqnSetup(), explore_chance=0.25),
    ]

    for i in tqdm(range(20), desc='Training AI', bar_format='{l_bar}{bar}'):
        play_many(game, players, 25)
        if i == 10:
            for player in players:
                player.explore_chance = 0.1

    for player in players:
        player.explore_chance = 0

    return game, players


if __name__ == '__main__':
    game1, players1 = get_sample_game_and_trained_players()
    play_human_ui(game1, players1)
