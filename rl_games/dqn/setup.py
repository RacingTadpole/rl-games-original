import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence, Callable, Generic, Any
from rl_games.core.game import State, Action, Game


@dataclass
class DqnSetup(Generic[State, Action]):
    num_states: int
    hidden_size: int
    num_actions: int
    get_input_vector: Callable[[State], np.ndarray]
    get_action_and_value_from_output: Callable[[Game, np.ndarray, Sequence[Action]], Tuple[Action, float]]
    get_onehot_index_from_action: Callable[[Game[State, Action], Action], int]
