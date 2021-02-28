# pylint: disable=unsubscriptable-object

from dataclasses import dataclass
from typing import Tuple
from .game import Game

@dataclass()
class DummyGame(Game[int, bool]):
    def get_actions(self):
        yield True
        yield False

    def update(self, action):
        if action is True:
            self.state += 1
        else:
            self.state -= 1

    def get_score_and_game_over(self) -> Tuple[int, bool]:
        if self.state == 5:
            return 1, True
        if self.state == -5:
            return 1, False
        return 0, False
