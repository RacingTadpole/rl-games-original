# pylint: disable=unsubscriptable-object

# Chopsticks

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, Sequence, cast, Generic, TypeVar
from copy import deepcopy
from .game import Game

FingerCount = int
HandIndex = int
PlayerIndex = int

PlayerState = Tuple[FingerCount, ...]

@dataclass(frozen=True)
class ChopsticksState:
    finger_counts: Tuple[PlayerState, ...] = ()
    next_turn: PlayerIndex = 0

@dataclass(frozen=True)
class ChopsticksAction:
    from_hand: HandIndex
    to_player: PlayerIndex
    to_hand: HandIndex
    number: FingerCount


@dataclass
class Chopsticks(Game[ChopsticksState, ChopsticksAction]):
    num_players: int = 2
    num_hands: int = 2

    def get_init_state(self, next_turn=0) -> ChopsticksState:
        """
        >>> game = Chopsticks()
        >>> game.get_init_state()
        ChopsticksState(finger_counts=((1, 1), (1, 1)), next_turn=0)
        """
        return ChopsticksState(finger_counts=((1,) * self.num_hands,) * self.num_players, next_turn=next_turn)

    def get_actions(self, state):
        """
        >>> game = Chopsticks()
        >>> state = game.get_init_state()
        >>> len(list(game.get_actions(state)))
        9
        >>> state = ChopsticksState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> list(game.get_actions(state))
        [(0, 1), (2, 0), (2, 1)]
        >>> state = ChopsticksState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> list(game.get_actions(state))
        [(2, 0), (2, 1)]
        """
        pass

    def updated(self, state: ChopsticksState, action: ChopsticksAction) -> ChopsticksState:
        """
        >>> game = Chopsticks()
        >>> state = game.get_init_state(next_turn=0)
        >>> game.updated(state, (2, 1))
        >>> state = ChopsticksState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.updated(state, (0, 1))
        """
        pass

    def _get_winner(self, state: ChopsticksState) -> None:
        """
        >>> game = Chopsticks()
        >>> state = ChopsticksState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game._get_winner(state)
        >>> state = ChopsticksState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game._get_winner(state)
        'O'
        """
        return None

    def get_score_and_game_over(self, state: ChopsticksState) -> Tuple[int, bool]:
        """
        >>> game = Chopsticks()
        >>> state = ChopsticksState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.get_score_and_game_over(state)
        (0, False)
        >>> state = ChopsticksState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game.get_score_and_game_over(state)  # O is the winner; X is the next player
        (1, True)
        >>> state = ChopsticksState(state.board, next_turn=o_marker)
        >>> game.get_score_and_game_over(state)
        (-1, True)
        """
        pass
