from abc import abstractmethod, ABC
from typing import Iterator, Generic, TypeVar, Tuple, NewType

State = TypeVar('State')
Action = TypeVar('Action')
PlayerIndex = NewType('PlayerIndex', int)

class Game(ABC, Generic[State, Action]):
    @abstractmethod
    def get_actions(self, state: State) -> Iterator[Action]:
        ...

    @abstractmethod
    def get_init_state(self) -> State:
        ...

    @abstractmethod
    def updated(self, state: State, action: Action) -> State:
        ...

    @abstractmethod
    def get_score_and_game_over(self, state: State) -> Tuple[float, bool]:
        """
        Return the reward for the player who took the most recent turn
        (ie. who got directly to this state).
        All other players get minus this reward.
        """
        ...
