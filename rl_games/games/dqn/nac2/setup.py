from dataclasses import dataclass
import numpy as np

from rl_games.games.nac import Nac, NacState, NacAction, empty_square
from rl_games.dqn.setup import (
    DqnSetup,
    GameAndStateToVector,
    GameAndStateToActionMask,
    OutputToActionAndValue,
    ActionToIndex,
    StateVector,
)
from ..nac.setup import get_nac_action_and_value_from_onehot_output, get_nac_action_mask, get_onehot_index_from_nac_action

# This experimental version of DQN NAC uses a non-one-hot state encoding,
# with just one state per position, with values -1 (X), 0 (empty) or 1 (O).


def get_nac_input(game: Nac, state: NacState) -> StateVector:
    """
    >>> game = Nac(size=2)
    >>> board = [[empty_square for _ in range(2)] for _ in range(2)]
    >>> get_nac_input(game, NacState(board, next_player_index=0))
    array([[0, 0, 0, 0]])
    >>> board[0][1] = game.markers[0]
    >>> get_nac_input(game, NacState(board, next_player_index=1))
    array([[ 0, -1,  0,  0]])
    >>> board[1][1] = game.markers[1]
    >>> get_nac_input(game, NacState(board, next_player_index=0))
    array([[ 0, -1,  0,  1]])
    """
    m = {empty_square: 0, game.markers[0]: -1, game.markers[1]: 1}
    return StateVector(np.array([m[e] for row in state.board for e in row], dtype=int).reshape(1, -1))


@dataclass
class NacDqnSetup(DqnSetup[NacState, NacAction]):
    num_states: int = 9
    hidden_size: int = 81
    num_actions: int = 9
    get_input_vector: GameAndStateToVector = get_nac_input
    get_action_mask: GameAndStateToActionMask = get_nac_action_mask
    get_action_and_value_from_output: OutputToActionAndValue = get_nac_action_and_value_from_onehot_output
    get_onehot_index_from_action: ActionToIndex = get_onehot_index_from_nac_action
