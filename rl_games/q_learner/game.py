# pylint: disable=unsubscriptable-object

import random
from dataclasses import dataclass, field
from typing import Iterator, Generic, TypeVar, Tuple
from copy import deepcopy

State = TypeVar('State')
Action = TypeVar('Action')

# Ideally this would be an abstract base class (abc) but mypy errors on dataclass abcs.
# TODO: I think this would be nicer as just Game, with class methods only.
@dataclass()
class GameState(Generic[State, Action]):
    state: State

    def get_actions(self) -> Iterator[Action]:
        pass

    def update(self, action: Action) -> None:
        """
        Updates state in-place.
        """
        pass

    def get_score_and_game_over(self) -> Tuple[int, bool]:
        pass
