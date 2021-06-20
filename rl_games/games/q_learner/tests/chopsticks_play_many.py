from typing import Sequence, Dict, Any

from rl_games.core.player import Player
from rl_games.core.play import play_many
from rl_games.games.chopsticks import Chopsticks


def chopsticks_play_many(
    players: Sequence[Player],
    *args: Any,
    **kwargs: Any
) -> Dict[str, float]:
    """
    >>> import random
    >>> from rl_games.q_learner.player import QPlayer
    >>> random.seed(2)
    >>> a, b = QPlayer('A'), QPlayer('B')
    >>> chopsticks_play_many([a, b], range(500))
    {'B': 0.524, 'A': 0.476}
    """
    game = Chopsticks()
    return play_many(game, players, *args, **kwargs)
