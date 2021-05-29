# pylint: disable=unsubscriptable-object

# Reinforcement Learning - Q training.
# Each player keeps a "Q table", ie. a mapping of (board, action) to values.
# The values are updated every turn using the Bellman equation.
import sys
import random
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Generic, Tuple, Literal, Optional, Iterator, Dict, List, Sequence
from collections import defaultdict

from .game import State, Action, Game


@dataclass
class Player(ABC, Generic[State, Action]):
    id: str = field(default_factory=lambda: f'{random.randrange(sys.maxsize)}')

    @abstractmethod
    def choose_action(self, game: Game[State, Action], state: State) -> Action:
        pass

    @abstractmethod
    def value(self, game: Game, state: State) -> float:
        pass

    @abstractmethod
    def update_action_value(
        self,
        game: Game,
        old_state: State,
        action: Action,
        new_state: State,
        reward: float,
    ) -> None:
        pass
