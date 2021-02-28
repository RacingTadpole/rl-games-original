# pylint: disable=unsubscriptable-object

from dataclasses import dataclass
from typing import Tuple
from .game import Game

class DummyGame(Game[int, bool]):
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
        if state == 4:
            return 1, True
        if state == -4:
            return 1, False
        return 0, False
