# Run with:
#     python -m rl_games.games.q_learner.chopsticks_play

from typing import Sequence, Tuple

from rl_games.q_learner.player import QPlayer
from rl_games.core.player import Player
from rl_games.core.play import play_many
from rl_games.core.play_human import play_human_ui
from rl_games.core.game import Game
from rl_games.games.chopsticks import Chopsticks, ChopsticksState, ChopsticksAction
from ..tqdm import range_with_timer


def get_sample_game_and_trained_players(num_rounds: int = 50000, initial_explore_chance: float = 0.25) -> Tuple[Game, Sequence[Player]]:
    game = Chopsticks()
    players = [
        QPlayer[ChopsticksState, ChopsticksAction]('P1', explore_chance=initial_explore_chance),
        QPlayer[ChopsticksState, ChopsticksAction]('P2', explore_chance=initial_explore_chance),
    ]

    play_many(game, players, range_with_timer(num_rounds), reduce_explore_chance=True)
    return game, players


if __name__ == '__main__':
    game1, players1 = get_sample_game_and_trained_players()
    play_human_ui(game1, players1)
