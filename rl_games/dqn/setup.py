from dataclasses import dataclass
from typing import Tuple, Sequence, Generic, Any
from typing_extensions import Protocol
import numpy as np

from rl_games.core.game import State, Action


class StateToVector(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__state: Any) -> np.ndarray: ...


class OutputToActionAndValue(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __output: np.ndarray, __actions: Sequence[Any]) -> Tuple[Any, float]: ...


class ActionToIndex(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __action: Any) -> int: ...


@dataclass
class DqnSetup(Generic[State, Action]):
    num_states: int
    hidden_size: int
    num_actions: int
    # Callable[[State], np.ndarray] would be simpler, but https://github.com/python/mypy/issues/5485
    get_input_vector: StateToVector
    # Callable[[Game, np.ndarray, Sequence[Action]], Tuple[Action, float]]
    get_action_and_value_from_output: OutputToActionAndValue
    # Callable[[Game[State, Action], Action], int]
    get_onehot_index_from_action: ActionToIndex
