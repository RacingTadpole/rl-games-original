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
    >>> from rl_games.games.nac import NacState, NacAction
    >>> from rl_games.dqn.dqn_player import DqnPlayer
    >>> from ..nac import NacDqnSetup
    >>> random.seed(2)
    >>> players = [DqnPlayer[NacState, NacAction]('X', NacDqnSetup()), DqnPlayer[NacState, NacAction]('O', NacDqnSetup())]
    >>> nac_play_many(players, 20)
    {'O': 0.05, 'X': 0.2}

    TODO: improve this test.
    """
    game = Nac()
    return play_many(game, players, *args, **kwargs)
