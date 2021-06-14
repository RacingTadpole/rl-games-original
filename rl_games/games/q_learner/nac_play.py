# Run with:
#     python -m rl_games.games.q_learner.nac_play

from typing import Sequence, Tuple
from rl_games.q_learner.player import QPlayer
from rl_games.core.player import Player
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.game import Game
from rl_games.games.nac import Nac, NacState, NacAction


def get_sample_game_and_trained_players() -> Tuple[Game, Sequence[Player]]:
    game = Nac()
    players = [QPlayer[NacState, NacAction]('X'), QPlayer[NacState, NacAction]('O')]

    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0.1
    play_many(game, players, 6000)
    for player in players:
        player.explore_chance = 0.05
    play_many(game, players, 4000)

    for player in players:
        player.explore_chance = 0

    return game, players


if __name__ == '__main__':
    print('Training AI...')
    game1, players1 = get_sample_game_and_trained_players()
    play_human_ui(game1, players1)
