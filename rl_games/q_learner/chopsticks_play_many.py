from typing import Sequence
import random

from .player import Player
from .play import play_many
from .chopsticks import Chopsticks


def chopsticks_play_many(
    players: Sequence[Player],
    *args,
    **kwargs
):
    """
    >>> random.seed(2)
    >>> a, b = Player('A'), Player('B')
    >>> chopsticks_play_many([a, b])
    {'A': 0.544, 'B': 0.456}
    """
    game = Chopsticks()
    return play_many(game, players, *args, **kwargs)
