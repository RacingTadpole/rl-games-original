from typing import Sequence
import random

from .player import Player
from .play import play_many
from .nac import Nac


def nac_play_many(
    players: Sequence[Player],
    *args,
    **kwargs
):
    """
    random.seed(2)
    x, o = Player('X'), Player('O')
    nac_play_many([x, o])
    {'X': 0.376, 'O': 0.185}
    """
    game = Nac()
    return play_many(game, players, *args, **kwargs)
