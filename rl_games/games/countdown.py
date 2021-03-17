# pylint: disable=unsubscriptable-object

from dataclasses import dataclass
from typing import Tuple
from rl_games.core.game import Game

@dataclass
class Countdown(Game[int, int]):
    """
    A simple game where the state is a number, and actions are numbers from 1 to 3.
    Start at 20 and when you hit 0, you win a point and end the game.
    """
    start: int = 20

    def get_actions(self, state):
        if state >= 3:
            yield 3
        if state >= 2:
            yield 2
        if state >= 1:
            yield 1

    def get_init_state(self) -> int:
        return self.start

    def updated(self, state, action):
        return state - action

    def get_score_and_game_over(self, state) -> Tuple[int, bool]:
        """
        In this game, if the state is 0, the last player to take a turn won.
        """
        if state == 0:
            return 1, True
        return 0, state < 0
