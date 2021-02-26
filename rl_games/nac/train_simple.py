# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - simple training.
# This records the history of moves and updates the value function at
# all visited positions at the end of every game, starting at the end and
# looping back through the positions with:
#     prior  = player.value[state]
#     reward = prior + player.learning_rate * (reward - prior)
#     player.value[state] = reward

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy

from .game import (
    Board, Player, Marker, get_init_board, get_updated_board, get_actions, get_other_marker, is_game_over, play_many
)


@dataclass
class SimplePlayer(Player):
    """
    >>> random.seed(2)
    >>> x, o = SimplePlayer(), SimplePlayer()
    >>> play_many(x, o, play_once=play_once_simple_training)
    (0.328, 0.269)

    Two experienced players usually come to a draw, with exploration turned off
    >>> x.explore_chance = 0
    >>> o.explore_chance = 0
    >>> play_once_simple_training(x, o, verbose=True)
    (('', '', ''), ('', '', ''), ('', 'X', ''))
    (('', '', 'O'), ('', '', ''), ('', 'X', ''))
    (('', '', 'O'), ('', 'X', ''), ('', 'X', ''))
    (('', 'O', 'O'), ('', 'X', ''), ('', 'X', ''))
    (('X', 'O', 'O'), ('', 'X', ''), ('', 'X', ''))
    (('X', 'O', 'O'), ('', 'X', ''), ('', 'X', 'O'))
    (('X', 'O', 'O'), ('', 'X', 'X'), ('', 'X', 'O'))
    (('X', 'O', 'O'), ('O', 'X', 'X'), ('', 'X', 'O'))
    (('X', 'O', 'O'), ('O', 'X', 'X'), ('X', 'X', 'O'))
    >>> play_many(x, o, play_once=play_once_simple_training)
    (0.166, 0.09)

    An experienced player against a novice should win
    >>> random.seed(2)
    >>> play_many(x, SimplePlayer(), play_once=play_once_simple_training)
    (0.288, 0.112)
    >>> play_many(x, SimplePlayer(), play_once=play_once_simple_training)
    (0.207, 0.01)

    Similarly for the player playing O, although this takes more training
    >>> random.seed(2)
    >>> o.explore_chance = 0.1
    >>> play_many(SimplePlayer(explore_chance=0.5), o, play_once=play_once_simple_training)
    (0.363, 0.43)
    >>> o.explore_chance = 0
    >>> play_many(SimplePlayer(), o, 100, play_once=play_once_simple_training)
    (0.17, 0.44)
    """

    _value: Dict[Board, float] = field(default_factory=dict)

    def value(self, board: Board, marker: Marker) -> float:
        return self._value.get(board, self.base_value)

    def update_values(
        self,
        states: List[Board],
        marker: Marker,
        final_reward: float,
    ) -> None:
        """
        Boards are only added to state after your turn, so for 'X',
        don't see 'O' moves.

        >>> states = [
        ...     (('X', '', ''), ('', '', ''), ('', '', '')),
        ...     (('X', '', ''), ('', 'O', ''), ('', '', 'X')),
        ...     (('X', '', 'O'), ('', 'O', ''), ('', '', 'X')),
        ...     (('X', '', 'O'), ('O', 'O', ''), ('X', '', 'X')),
        ... ]
        >>> player = SimplePlayer()
        >>> player.update_values(states, 'X', 1)
        >>> for k,v in player._value.items():
        ...     print(f'{k}: {v}')
        (('X', '', 'O'), ('O', 'O', ''), ('X', '', 'X')): 0.1
        (('X', '', 'O'), ('', 'O', ''), ('', '', 'X')): 0.01
        (('X', '', ''), ('', 'O', ''), ('', '', 'X')): 0.001
        (('X', '', ''), ('', '', ''), ('', '', '')): 0.0001
        """
        reward = final_reward
        for state in states[-1::-1]:
            prior = self.value(state, marker)
            reward = prior + self.learning_rate * (reward - prior)
            self._value[state] = round(reward, 5)

def play_once_simple_training(
    player_x: SimplePlayer,
    player_o: SimplePlayer,
    verbose = False,
    restrict_opening = False,
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    >>> random.seed(1)
    >>> x, o = SimplePlayer(), SimplePlayer()
    >>> play_once_simple_training(x, o)
    'X'
    >>> for k, v in x._value.items():
    ...         print(f'X: {k}: {v}')
    X: (('O', 'X', 'X'), ('O', 'X', 'O'), ('X', 'X', 'O')): 1
    X: (('', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): 0.1
    X: (('', '', 'X'), ('', '', 'O'), ('X', 'X', 'O')): 0.01
    X: (('', '', ''), ('', '', ''), ('X', 'X', 'O')): 0.001
    X: (('', '', ''), ('', '', ''), ('X', '', '')): 0.0001
    >>> for k, v in o._value.items():
    ...         print(f'O: {k}: {v}')
    O: (('O', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): -0.1
    O: (('', '', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): -0.01
    O: (('', '', ''), ('', '', 'O'), ('X', 'X', 'O')): -0.001
    O: (('', '', ''), ('', '', ''), ('X', '', 'O')): -0.0001
    """
    board = get_init_board()
    history: Dict[Marker, List[Board]] = {'X': [], 'O': []}
    players = {'X': player_x, 'O': player_o}
    score = 0
    game_over = False
    while not game_over:
        for marker, player in players.items():
            marker = cast(Marker, marker)
            action = player.choose_action(board, marker, restrict_opening)
            board = get_updated_board(board, action)
            if verbose:
                print(board)
            history[marker].append(board)
            game_over, score = is_game_over(board, marker)
            if game_over:
                break

    marker = cast(Marker, marker)
    # Override the learning rate on the final board reward - just set it to the score. (Optional.)
    player._value[board] = score
    player.update_values(history[marker][:-1], marker, score)

    other_marker = get_other_marker(marker)
    other_player = players[other_marker]
    other_player.update_values(history[other_marker], other_marker, -score)

    if score > 0:
        return marker
    if score < 0:
        return other_marker
    return None
