import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Sequence, Callable
from collections import defaultdict

from rl_games.core.game import State, Action, Game
from rl_games.core.player import Player
from rl_games.games.nac import Nac, NacState, NacAction, x_marker, o_marker, empty_square
from rl_games.neural.neural_network import NeuralNetwork
from rl_games.dqn.onehot import get_onehot_vector_from_index
from rl_games.dqn.setup import DqnSetup, StateToVector, OutputToActionAndValue, ActionToIndex


m = {empty_square: 0, x_marker: 1, o_marker: 2}


def get_nac_state_index(state: NacState) -> int:
    """
    Index states as:
        sum over row r, column c of m(r, c) * 3 ^ i(r, c)
    where
        m(r, c) = 0 if empty, or 1, 2 for players
        i(r, c) = r * size + c
    Then, if 'o' is next player, add 3 ^ size^2
    
    >>> board = [[empty_square,] * 3,] * 3
    >>> board
    [['', '', ''], ['', '', ''], ['', '', '']]
    >>> get_nac_state_index(NacState(board, x_marker))
    0
    >>> get_nac_state_index(NacState(board, o_marker))
    19683
    >>> board[0][1] = x_marker
    >>> get_nac_state_index(NacState(board, x_marker))
    2271
    >>> board[1][1] = o_marker
    >>> get_nac_state_index(NacState(board, x_marker))
    4542
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


def get_nac_action_and_value_from_onehot_output(game: Nac, output: np.ndarray, legal_actions: Sequence[NacAction]) -> Tuple[NacAction, float]:
    size = game.size
    mask_to_clear = np.ones((size, size), dtype=bool)
    for action in legal_actions:
        mask_to_clear[action.row][action.col] = False
    # TODO: output the legal actions as a mask in the first place
    masked_output = output.copy()
    masked_output[mask_to_clear.reshape(masked_output.shape)] = -1
    max_index = np.argmax(masked_output)
    max_value = masked_output[0][max_index]
    return NacAction(row=int(max_index // size), col=int(max_index % size)), max_value


def get_onehot_index_from_nac_action(game: Nac, action: NacAction) -> int:
    return action.row * game.size + action.col


@dataclass
class NacDqnSetup(DqnSetup[NacState, NacAction]):
    num_states: int = 3 ** 9 * 2
    hidden_size: int = 18
    num_actions: int = 9
    get_input_vector: StateToVector = get_onehot_nac_input
    get_action_and_value_from_output: OutputToActionAndValue = get_nac_action_and_value_from_onehot_output
    get_onehot_index_from_action: ActionToIndex = get_onehot_index_from_nac_action


