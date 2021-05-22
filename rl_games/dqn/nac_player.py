# pylint: disable=unsubscriptable-object

# Reinforcement Learning - deep q network.
# Each player uses a neural network model.

# To start with, let's just implement a noughts-and-crosses player here.
# The state is:  board: Tuple[Tuple[Square, ...], ...]; next_turn: Marker = x_marker
# For simplicitly let's encode the board as size^2 squares, each of which could be any of 3 states,
# and the next turn as either player, ie. (size * size * 3) * 2.
# A NAC action is just a position on the board, ie. size * size.
# For simplicity let's assume size = 3, ie. 54 states and 9 actions.

import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Generic, Tuple, Literal, Optional, Iterator, Dict, List, Sequence, Any, Callable
from collections import defaultdict

from rl_games.core.game import State, Action, Game
from rl_games.neural.neural_network import NeuralNetwork
from rl_games.games.nac import NacState, x_marker, o_marker, empty_square


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

def get_onehot_input_from_index(index: int, size: int) -> np.ndarray:
    """
    >>> get_onehot_input_from_index(0, 5)
    array([1, 0, 0, 0, 0])
    >>> get_onehot_input_from_index(3, 4)
    array([0, 0, 0, 1])
    """
    x = np.zeros(size, dtype=int)
    x[index] = 1
    return x.reshape(1, size) # np.array([1 if i == index else 0 for i in range(size)])

def get_onehot_nac_input(state: NacState) -> np.ndarray:
    """
    >>> board = [[empty_square,] * 2,] * 2
    >>> s = get_onehot_nac_input(NacState(board, x_marker))
    >>> len(s), s[:5]
    (162, array([1, 0, 0, 0, 0]))
    >>> board[0][1] = x_marker
    >>> s = get_onehot_nac_input(NacState(board, o_marker))
    >>> s[:5], s[81:86]
    (array([0, 0, 0, 0, 0]), array([1, 0, 0, 0, 0]))
    """
    size = 2 * 3 ** len(state.board) ** 2
    return get_onehot_input_from_index(get_nac_state_index(state), size)

@dataclass
class Player(Generic[State, Action]):
    id: str = field(default_factory=lambda: f'{random.randrange(sys.maxsize)}')
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    discount_factor: float = 0.9

    # These defaults are for NAC
    num_states: int = 3 ** 9 * 2
    hidden_size: int = 18
    num_actions: int = 9
    get_input_vector: Callable[[NacState], np.ndarray] = get_onehot_nac_input  # TODO: [State]

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = NeuralNetwork(
            input_size=self.num_states,
            hidden_size=self.hidden_size,
            output_size=self.num_actions
        )

    def choose_action(self, game: Game[State, Action], state: State) -> Action:
        """
        >>> from rl_games.games.countdown import Countdown
        >>> random.seed(3)
        >>> game = Countdown()
        >>> player = Player[int, int]('A', explore_chance=0)
        >>> player.choose_action(game, 5), player.choose_action(game, 5)
        (2, 3)
        """
        actions = list(game.get_actions(state))
        if random.uniform(0, 1) <= self.explore_chance:
            # Explore
            return random.choice(actions)
        # Greedy action - choose action with greatest expected value
        # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
        random.shuffle(actions)
        if len(actions) == 0:
            raise IndexError(f'No actions available from {state}')
        best_action_index = np.argmax(self.model.predict(self.get_input_vector(state)))
        return actions[best_action_index]

    def value(self, game: Game, state: State) -> float:
        """
        >>> from rl_games.games.countdown import Countdown
        >>> random.seed(2)
        >>> game = Countdown()
        >>> player = Player[int, int]() # action_value={(1, 1): 1, (1, 2): 0, (2, 3): -7, (3, 3): 2})
        >>> player.value(game, 1), player.value(game, 2), player.value(game, 3)
        (1, 1, 1)
        """
        actions = list(game.get_actions(state))
        if len(actions):
            return 1 # max(self.action_value.get((state, action), 0) for action in actions)
        # If no actions are possible, the game must be over, and the value is 0.
        return 0

    def update_action_value(
        self,
        game: Game,
        old_state: State,
        action: Action,
        new_state: State,
        reward: float,
    ) -> None:
        """
        Note that new_state is the next state in which this player can move again,
        ie. it includes opponent moves.
        We'll imagine the dummy game is a 2-player game.
        >>> from rl_games.games.countdown import Countdown
        >>> random.seed(2)
        >>> game = Countdown()
        >>> player = Player()
        >>> player.update_action_value(game, 4, True, 6, 1)
        >>> player.update_action_value(game, 2, True, 4, 0)
        >>> player.update_action_value(game, 0, True, 2, 0)
        """
        pass
        # self.action_value[old_state, action] += reward + self.learning_rate * (
        #     self.discount_factor * self.value(game, new_state)
        #     - self.action_value[old_state, action])
