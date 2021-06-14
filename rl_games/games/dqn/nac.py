from dataclasses import dataclass
from typing import Tuple, Sequence
import numpy as np

from rl_games.games.nac import Nac, NacState, NacAction, x_marker, o_marker, empty_square
from rl_games.dqn.onehot import get_onehot_vector_from_index
from rl_games.dqn.setup import (
    DqnSetup, StateToVector, GameAndStateToActionMask, OutputToActionAndValue, ActionToIndex
)


m = {empty_square: 0, x_marker: 1, o_marker: 2}


def get_nac_state_index(state: NacState) -> int:
    """
    Index states as:
        sum over row r, column c of m(r, c) * 3 ^ i(r, c)
    where
        m(r, c) = 0 if empty, or 1, 2 for players
        i(r, c) = r * size + c
    Then, if 'o' is next player, add 3 ^ size^2

    >>> board = [[empty_square for _ in range(3)] for _ in range(3)]
    >>> board
    [['', '', ''], ['', '', ''], ['', '', '']]
    >>> get_nac_state_index(NacState(board, x_marker))
    0
    >>> get_nac_state_index(NacState(board, o_marker))
    19683
    >>> board[0][1] = x_marker
    >>> get_nac_state_index(NacState(board, x_marker))
    3
    >>> board[1][1] = o_marker
    >>> get_nac_state_index(NacState(board, x_marker))
    165
    """
    one_d = [m[marker] for r, row in enumerate(state.board) for marker in row]
    board_index = sum(v * 3 ** i for i, v in enumerate(one_d))
    return board_index + 0 if state.next_turn == x_marker else 3 ** len(one_d)


def get_onehot_nac_input(state: NacState) -> np.ndarray:
    """
    >>> board = [[empty_square,] * 2,] * 2
    >>> s = get_onehot_nac_input(NacState(board, x_marker))
    >>> s.shape, s[:, :5]
    ((1, 162), array([[1, 0, 0, 0, 0]]))
    >>> board[0][1] = x_marker
    >>> s = get_onehot_nac_input(NacState(board, o_marker))
    >>> s[:, :5], s[:, 81:86]
    (array([[0, 0, 0, 0, 0]]), array([[1, 0, 0, 0, 0]]))
    """
    size = 2 * 3 ** len(state.board) ** 2
    return get_onehot_vector_from_index(get_nac_state_index(state), size)


def get_nac_action_mask(game: Nac, state: NacState) -> np.ndarray:
    """
    Returns a mask where TRUE means NOT valid, in line with numpy's masked array type,
    https://numpy.org/doc/stable/reference/maskedarray.baseclass.html

    >>> game = Nac(size = 3)
    >>> board = [[empty_square for _ in range(3)] for _ in range(3)]
    >>> board[0][1] = x_marker
    >>> state = NacState(board, o_marker)
    >>> get_nac_action_mask(game, state)
    array([[False,  True, False],
           [False, False, False],
           [False, False, False]])
    """
    size = game.size
    return np.array([[state.board[r][c] != empty_square for c in range(size)] for r in range(size)])


def get_nac_action_and_value_from_onehot_output(
    game: Nac,
    output: np.ndarray,
    action_mask: np.ndarray,
) -> Tuple[NacAction, float]:
    """
    >>> game = Nac(size = 3)
    >>> mask = np.array([[False,  True, False], [False, False, False], [False, False, False]])
    >>> output = np.array([[1.1, 20, 3], [4, 5, 6], [7, 8, 6.5]])
    >>> get_nac_action_and_value_from_onehot_output(game, output, mask)
    ((2, 1), 8.0)

    In the above, the max is 20 at (0, 1), but this position is excluded by the mask.
    """
    size = game.size
    masked_output = np.ma.array(output, mask=action_mask, fill_value=-np.inf, dtype=np.float16)
    max_flattened_index = np.ma.argmax(masked_output)
    # TODO: extract this conversion into another function too.
    row = int(max_flattened_index // size)
    col = int(max_flattened_index % size)
    max_value = masked_output[row][col]
    return NacAction(row, col), max_value


def get_onehot_index_from_nac_action(game: Nac, action: NacAction) -> int:
    return action.row * game.size + action.col


@dataclass
class NacDqnSetup(DqnSetup[NacState, NacAction]):
    num_states: int = 3 ** 9 * 2
    hidden_size: int = 18
    num_actions: int = 9
    get_input_vector: StateToVector = get_onehot_nac_input
    get_action_mask: GameAndStateToActionMask = get_nac_action_mask
    get_action_and_value_from_output: OutputToActionAndValue = get_nac_action_and_value_from_onehot_output
    get_onehot_index_from_action: ActionToIndex = get_onehot_index_from_nac_action
