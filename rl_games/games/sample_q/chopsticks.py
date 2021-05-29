# pylint: disable=unsubscriptable-object

# Run with:
#     python -m rl_games.games.sample_q.chopsticks

from typing import Union, Sequence, Tuple
from rl_games.q_learner.player import QPlayer
from rl_games.core.player import Player
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.game import Game
from rl_games.games.chopsticks import Chopsticks, ChopsticksState, ChopsticksAction


def get_sample_game_and_trained_players() -> Tuple[Game, Sequence[Player]]:
    game = Chopsticks()
    players = [QPlayer[ChopsticksState, ChopsticksAction]('P1'), QPlayer[ChopsticksState, ChopsticksAction]('P2')]

    play_many(game, players, 25000)
    for player in players:
        player.explore_chance = 0.1
    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0.05
    play_many(game, players, 9000)

    for player in players:
        player.explore_chance = 0

    return game, players


if __name__ == '__main__':
    print('Training AI...')
    game, players = get_sample_game_and_trained_players()
    play_human_ui(game, players)
