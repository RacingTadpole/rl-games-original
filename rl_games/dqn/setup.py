from dataclasses import dataclass
from typing import Tuple, Generic, Any
from typing_extensions import Protocol

from rl_games.core.game import State, Action
from rl_games.dqn.types import ActionMask, ActionVector, StateVector

class GameAndStateToVector(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __state: Any) -> StateVector: ...


class GameAndStateToActionMask(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __state: Any) -> ActionMask: ...

class OutputToActionAndValue(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __output: ActionVector, __action_mask: ActionMask) -> Tuple[Any, float]: ...


class ActionToIndex(Protocol):
    # pylint: disable=too-few-public-methods, invalid-name
    @staticmethod
    def __call__(__game: Any, __action: Any) -> int: ...


@dataclass
class DqnSetup(Generic[State, Action]):
    num_states: int
    hidden_size: int
    num_actions: int

    # Callable[[Game, State], StateVector] would be simpler, but https://github.com/python/mypy/issues/5485
    get_input_vector: GameAndStateToVector

    # Callable[[Game, State], ActionMask]
    get_action_mask: GameAndStateToActionMask

    # Callable[[Game, ActionVector, ActionMask], Tuple[Action, float]]
    get_action_and_value_from_output: OutputToActionAndValue

    # Callable[[Game[State, Action], Action], int]
    get_onehot_index_from_action: ActionToIndex
