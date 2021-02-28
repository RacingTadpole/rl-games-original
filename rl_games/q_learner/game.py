# pylint: disable=unsubscriptable-object

import random
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Iterator, Generic, TypeVar, Tuple
from copy import deepcopy

State = TypeVar('State')
Action = TypeVar('Action')

class Game(ABC, Generic[State, Action]):
    @abstractmethod
    def get_actions(self, state: State) -> Iterator[Action]:
        pass

    @abstractmethod
    def get_init_state(self) -> State:
        pass

    @abstractmethod
    def updated(self, state: State, action: Action) -> State:
        pass

    @abstractmethod
    def get_score_and_game_over(self, state: State) -> Tuple[int, bool]:
        pass
