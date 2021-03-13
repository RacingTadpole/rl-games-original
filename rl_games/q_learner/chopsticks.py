# pylint: disable=unsubscriptable-object

# Chopsticks

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, Sequence, cast, Generic, TypeVar
from copy import deepcopy
from .game import Game

MAX_ROUNDS = 1000

FingerCount = int
HandIndex = int
PlayerIndex = int

PlayerState = Tuple[FingerCount, ...]

@dataclass(frozen=True)
class ChopsticksState:
    finger_counts: Tuple[PlayerState, ...] = ()
    next_turn: PlayerIndex = 0
    num_rounds: int = 0

    def __str__(self) -> str:
        return '\n'.join([f'P{i + 1}: {fingers}' for i, fingers in enumerate(self.finger_counts)])

@dataclass(frozen=True, repr=False)
class ChopsticksAction:
    from_hand: HandIndex
    to_player: PlayerIndex
    to_hand: HandIndex
    fingers: FingerCount

    def __repr__(self) -> str:
        return f'H{self.from_hand + 1} {self.fingers} -> H{self.to_hand + 1} P{self.to_player + 1}'


@dataclass
class Chopsticks(Game[ChopsticksState, ChopsticksAction]):
    num_players: int = 2
    num_hands: int = 2
    fingers_per_hand: int = 4

    def get_init_state(self, next_turn=0) -> ChopsticksState:
        """
        >>> game = Chopsticks()
        >>> game.get_init_state()
        ChopsticksState(finger_counts=((1, 1), (1, 1)), next_turn=0, num_rounds=0)
        """
        return ChopsticksState(finger_counts=((1,) * self.num_hands,) * self.num_players, next_turn=next_turn)

    def get_actions(self, state: ChopsticksState):
        """
        >>> game = Chopsticks()
        >>> state = game.get_init_state()
        >>> len(list(game.get_actions(state)))
        6
        >>> state = ChopsticksState(((0, 1), (2, 0)), 0)
        >>> list(game.get_actions(state))
        [H2 1 -> H1 P2]
        >>> state = ChopsticksState(((1, 1), (2, 0)), 0)
        >>> list(game.get_actions(state))
        [H1 1 -> H2 P1, H2 1 -> H1 P1, H1 1 -> H1 P2, H2 1 -> H1 P2]
        >>> state = ChopsticksState(((1, 1), (2, 0)), 1)
        >>> list(game.get_actions(state))
        [H1 1 -> H1 P1, H1 1 -> H2 P1, H1 2 -> H1 P1, H1 2 -> H2 P1, H1 1 -> H2 P2]
        """
        this_player = state.next_turn
        for to_player in range(self.num_players):
            for from_hand in range(self.num_hands):
                num_fingers = state.finger_counts[this_player][from_hand]
                for fingers in range(1, 1 + num_fingers):
                    for to_hand in range(self.num_hands):
                        # Special rules when giving to yourself:
                        # 1. You can't transfer to the same hand (of course).
                        # 2. You can't leave the set of finger counts unchanged.
                        if to_player == this_player:
                            if from_hand == to_hand:
                                continue
                            if set(state.finger_counts[this_player]) == {
                                x if i not in [from_hand, to_hand] else (x - fingers if i == from_hand else x + fingers)
                                for i, x in enumerate(state.finger_counts[this_player])
                            }:
                                continue
                        # Special rules when hitting another player:
                        # 1. You can't hit a 0-finger hand.
                        else:
                            if state.finger_counts[to_player][to_hand] == 0:
                                continue

                        yield ChopsticksAction(from_hand, to_player, to_hand, fingers)


    def updated(self, state: ChopsticksState, action: ChopsticksAction) -> ChopsticksState:
        """
        >>> game = Chopsticks()
        >>> state = game.get_init_state(next_turn=0)
        >>> actions = list(game.get_actions(state))
        >>> actions[0], actions[2]
        (H1 1 -> H2 P1, H1 1 -> H1 P2)
        >>> game.updated(state, actions[0])
        ChopsticksState(finger_counts=((0, 2), (1, 1)), next_turn=1, num_rounds=1)
        >>> game.updated(state, actions[2])
        ChopsticksState(finger_counts=((1, 1), (2, 1)), next_turn=1, num_rounds=1)
        >>> state = ChopsticksState(finger_counts=((4, 1), (1, 1)), next_turn=1)
        >>> game.updated(state, ChopsticksAction(from_hand=0, to_player=0, to_hand=0, fingers=1))
        ChopsticksState(finger_counts=((0, 1), (1, 1)), next_turn=0, num_rounds=1)
        """
        this_player = state.next_turn
        updated_counts = [list(player_state) for player_state in state.finger_counts]
        updated_counts[action.to_player][action.to_hand] += action.fingers
        if updated_counts[action.to_player][action.to_hand] > self.fingers_per_hand:
            updated_counts[action.to_player][action.to_hand] = 0
        if this_player == action.to_player:
            updated_counts[this_player][action.from_hand] -= action.fingers
        return ChopsticksState(
            finger_counts=tuple(tuple(player_state) for player_state in updated_counts),
            next_turn=(this_player + 1) % self.num_players,
            num_rounds=state.num_rounds + 1
        )

    def get_score_and_game_over(self, state: ChopsticksState) -> Tuple[int, bool]:
        """
        In this game, you only win if you knock everyone else down to all zero fingers.
        In this case, you must have been the last player to take a turn.
        We also cap the game at 1000 turns.
        >>> game = Chopsticks()
        >>> state = ChopsticksState(finger_counts=((4, 1), (1, 1)), next_turn=1, num_rounds=0)
        >>> game.get_score_and_game_over(state)
        (0, False)
        >>> state = ChopsticksState(finger_counts=((0, 0), (1, 1)), next_turn=0, num_rounds=0)
        >>> game.get_score_and_game_over(state)
        (1, True)
        """
        has_lost = [all(fingers == 0 for fingers in player_state) for player_state in state.finger_counts]
        if sum(1 if l else 0 for l in has_lost) == 1:
            return 1, True
        return 0, state.num_rounds > MAX_ROUNDS
