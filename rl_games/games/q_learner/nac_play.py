# Run with:
#     python -m rl_games.games.q_learner.nac_play

from typing import Sequence, Tuple
from tqdm import tqdm

from rl_games.q_learner.player import QPlayer
from rl_games.core.player import Player
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.game import Game
from rl_games.games.nac import Nac, NacState, NacAction


def get_sample_game_and_trained_players() -> Tuple[Game, Sequence[Player]]:
    game = Nac()
    players = [
        QPlayer[NacState, NacAction]('X', explore_chance=0.2),
        QPlayer[NacState, NacAction]('O', explore_chance=0.2),
    ]

    for i in tqdm(range(30), desc='Training AI', bar_format='{l_bar}{bar}'):
        play_many(game, players, 1000)
        if i in (15, 25):
            for player in players:
                player.explore_chance = (i == 9 and 0.1 or 0.05)

    for player in players:
        player.explore_chance = 0

    return game, players


if __name__ == '__main__':
    game1, players1 = get_sample_game_and_trained_players()
    play_human_ui(game1, players1)
