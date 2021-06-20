from typing import Sequence, Dict, Any

from rl_games.q_learner.player import Player
from rl_games.core.play import play_many
from rl_games.games.nac import Nac


def nac_play_many(
    players: Sequence[Player],
    *args: Any,
    **kwargs: Any
) -> Dict[str, float]:
    """
    >>> import random
    >>> from rl_games.q_learner.player import QPlayer
    >>> random.seed(2)
    >>> x, o = QPlayer('X'), QPlayer('O')
    >>> nac_play_many([x, o], range(500))
    {'X': 0.46, 'O': 0.208}
    """
    game = Nac()
    return play_many(game, players, *args, **kwargs)
