# pylint: disable=unsubscriptable-object

from dataclasses import dataclass
from typing import Tuple
from .game import Game

@dataclass
class DummyGame(Game[int, bool]):
    """
    A simple game where the state is a number, and actions are True to add 1, False to subtract 1.
    Start at 0 and when you hit the target, you win a point and end the game.
    """
    target: int = 6

    def get_actions(self, state):
        yield True
        yield False

    def get_init_state(self) -> int:
        return 0

    def updated(self, state, action):
        if action is True:
            return state + 1
        else:
            return state - 1

    def get_score_and_game_over(self, state) -> Tuple[int, bool]:
        if state == self.target:
            return 1, True
        return 0, False
