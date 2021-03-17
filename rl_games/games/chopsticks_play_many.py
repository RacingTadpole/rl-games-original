from typing import Sequence
import random

from q_learner.player import Player
from core.play import play_many
from games.chopsticks import Chopsticks


def chopsticks_play_many(
    players: Sequence[Player],
    *args,
    **kwargs
):
    """
    >>> random.seed(2)
    >>> a, b = Player('A'), Player('B')
    >>> chopsticks_play_many([a, b])
    {'B': 0.514, 'A': 0.486}
    """
    game = Chopsticks()
    return play_many(game, players, *args, **kwargs)
