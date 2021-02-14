# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - simple training.
# This records the history of moves and updates the value function at all visited points on every game, using:
#     reward = prior + player.learning_rate * (reward - prior)

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy

from .game import (
    Board, Player, Marker, get_init_board, choose_action, get_updated_board, get_actions, get_other_marker, is_game_over, play_many
)

"""
Returns the fraction won by player x and o.
>>> random.seed(2)
>>> x, o = Player(), Player()
>>> play_many(x, o, play_once=play_once_simple_training)
(0.328, 0.269)

Two experienced players usually come to a draw, with exploration turned off
>>> x.explore_chance = 0
>>> o.explore_chance = 0
>>> play_once(x, o, verbose=True)
((None, None, None), (None, None, None), (None, 'X', None))
((None, None, 'O'), (None, None, None), (None, 'X', None))
((None, None, 'O'), (None, 'X', None), (None, 'X', None))
((None, 'O', 'O'), (None, 'X', None), (None, 'X', None))
(('X', 'O', 'O'), (None, 'X', None), (None, 'X', None))
(('X', 'O', 'O'), (None, 'X', None), (None, 'X', 'O'))
(('X', 'O', 'O'), (None, 'X', 'X'), (None, 'X', 'O'))
(('X', 'O', 'O'), ('O', 'X', 'X'), (None, 'X', 'O'))
(('X', 'O', 'O'), ('O', 'X', 'X'), ('X', 'X', 'O'))
>>> play_many(x, o, play_once=play_once_simple_training)
(0.166, 0.09)

An experienced player against a novice should win
>>> random.seed(2)
>>> play_many(x, Player(), play_once=play_once_simple_training)
(0.288, 0.112)
>>> play_many(x, Player(), play_once=play_once_simple_training)
(0.207, 0.01)

Similarly for the player playing O, although this takes more training
>>> random.seed(2)
>>> o.explore_chance = 0.1
>>> play_many(Player(explore_chance=0.5), o, play_once=play_once_simple_training)
(0.363, 0.43)
>>> o.explore_chance = 0
>>> play_many(Player(), o, 100, play_once=play_once_simple_training)
(0.17, 0.44)
"""

def update_values(
    states: List[Board],
    player: Player,
    final_reward: float,
) -> None:
    """
    Boards are only added to state after your turn, so for 'X',
    don't see 'O' moves.

    >>> states = [
    ...     (('X', None, None), (None, None, None), (None, None, None)),
    ...     (('X', None, None), (None, 'O', None), (None, None, 'X')),
    ...     (('X', None, 'O'), (None, 'O', None), (None, None, 'X')),
    ...     (('X', None, 'O'), ('O', 'O', None), ('X', None, 'X')),
    ... ]
    >>> player = Player()
    >>> update_values(states, player, 1)
    >>> for k,v in player.value.items():
    ...     print(f'{k}: {v}')
    (('X', None, 'O'), ('O', 'O', None), ('X', None, 'X')): 0.1
    (('X', None, 'O'), (None, 'O', None), (None, None, 'X')): 0.01
    (('X', None, None), (None, 'O', None), (None, None, 'X')): 0.001
    (('X', None, None), (None, None, None), (None, None, None)): 0.0001
    """
    reward = final_reward
    for state in states[-1::-1]:
        prior = player.value.get(state, player.base_value)
        reward = prior + player.learning_rate * (reward - prior)
        player.value[state] = round(reward, 5)

def play_once_simple_training(
    player_x: Player,
    player_o: Player,
    verbose = False
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    >>> random.seed(1)
    >>> x, o = Player(), Player()
    >>> play_once_simple_training(x, o)
    'X'
    >>> for k, v in x.value.items():
    ...         print(f'X: {k}: {v}')
    X: (('O', 'X', 'X'), ('O', 'X', 'O'), ('X', 'X', 'O')): 1
    X: ((None, 'X', 'X'), ('O', None, 'O'), ('X', 'X', 'O')): 0.1
    X: ((None, None, 'X'), (None, None, 'O'), ('X', 'X', 'O')): 0.01
    X: ((None, None, None), (None, None, None), ('X', 'X', 'O')): 0.001
    X: ((None, None, None), (None, None, None), ('X', None, None)): 0.0001
    >>> for k, v in o.value.items():
    ...         print(f'O: {k}: {v}')
    O: (('O', 'X', 'X'), ('O', None, 'O'), ('X', 'X', 'O')): -0.1
    O: ((None, None, 'X'), ('O', None, 'O'), ('X', 'X', 'O')): -0.01
    O: ((None, None, None), (None, None, 'O'), ('X', 'X', 'O')): -0.001
    O: ((None, None, None), (None, None, None), ('X', None, 'O')): -0.0001
    """
    board = get_init_board()
    history: Dict[Marker, List[Board]] = {'X': [], 'O': []}
    players = {'X': player_x, 'O': player_o}
    score = 0
    game_over = False
    while not game_over:
        for marker, player in players.items():
            marker = cast(Marker, marker)
            action = choose_action(board, player, marker)
            board = get_updated_board(board, action, marker)
            if verbose:
                print(board)
            history[marker].append(board)
            game_over, score = is_game_over(board, marker)
            if game_over:
                break

    marker = cast(Marker, marker)
    # Override the learning rate on the final board reward - just set it to the score. (Optional.)
    player.value[board] = score
    update_values(history[marker][:-1], player, score)

    other = get_other_marker(marker)
    update_values(history[other], players[other], -score)

    if score > 0:
        return marker
    if score < 0:
        return other
    return None
