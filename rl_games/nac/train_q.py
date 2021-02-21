# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - Q training.
# Each player keeps a "Q table", ie. a mapping of (board, action) to values.
# The values are updated every turn using the Bellman equation.

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy
from collections import defaultdict

from .game import (
    Board, Player, Marker, Action,
    get_init_board, get_updated_board, get_actions,
    get_other_marker, is_game_over, play_many
)


@dataclass
class QPlayer(Player):
    """
    >>> random.seed(2)
    >>> x, o = QPlayer(explore_chance=0.4), QPlayer(explore_chance=0.4)
    >>> a = play_many(x, o, 20000, play_once=play_once_q_training, restrict_opening=True)

    >>> x.explore_chance = 0
    >>> o.explore_chance = 0
    >>> a = play_many(x, o, 10000, play_once=play_once_q_training, restrict_opening=True)

    >>> x2 = QPlayer()
    >>> play_many(x2, o, 10000, play_once=play_once_q_training, restrict_opening=True)
    (0.582, 0.2847)

    >>> # x.print_values()
    >>> o.print_values()
    """

    # Note the defaultdict defaults the action_value to 0, not to self.base_value.
    action_value: Dict[Tuple[Board, Action], float] = field(default_factory=lambda: defaultdict(float))
    discount_factor: float = 0.9

    def value(self, board: Board, marker: Marker) -> float:
        actions = list(get_actions(board, marker))
        if len(actions):
            return max(self.action_value.get((board, action), self.base_value)
                       for action in actions)
        # If no actions are possible, the game must be over, and the value is 0.
        return 0
    
    def print_action_values(self):
        for (board, action), value in sorted(self.action_value.items()):
            print(f'{board}: {action} = {value:.5f}')

    def print_non_zero_action_values(self):
        for (board, action), value in sorted(self.action_value.items(), key=lambda item: (-item[1], item[0])):
            if value != 0:
                print(f'{board}: {action} = {value:.5f}')

    def print_values(self):
        for board in sorted({k[0]: 1 for k in self.action_value.keys()}.keys()):
            x = self.value(board, 'X')
            o = self.value(board, 'O')
            print(f'{board}: X={x:.5f} O={o:.5f}')

    def update_action_value(
        self,
        old_board: Board,
        action: Action,
        new_board: Board,
        reward: float,
    ) -> '':
        """
        >>> boards = [
        ...     (('', '', ''), ('', '', ''), ('', '', '')),
        ...     (('X', '', ''), ('', 'O', ''), ('', '', '')),
        ...     (('X', '', 'O'), ('', 'O', ''), ('', '', 'X')),
        ...     (('X', '', 'O'), ('O', 'O', ''), ('X', '', 'X')),
        ...     (('X', '', 'O'), ('O', 'O', ''), ('X', 'X', 'X')),
        ... ]
        >>> player = QPlayer()
        >>> player.update_action_value(boards[-2], (2, 1, 'X'), boards[-1], 1)
        >>> player.update_action_value(boards[-3], (2, 0, 'X'), boards[-2], 0)
        >>> player.update_action_value(boards[-4], (2, 2, 'X'), boards[-3], 0)
        >>> player.print_action_values()
        (('X', '', ''), ('', 'O', ''), ('', '', '')): (2, 2, 'X') = 0.00810
        (('X', '', 'O'), ('', 'O', ''), ('', '', 'X')): (2, 0, 'X') = 0.09000
        (('X', '', 'O'), ('O', 'O', ''), ('X', '', 'X')): (2, 1, 'X') = 1.00000
        """
        marker = action[-1]
        self.action_value[old_board, action] += reward + self.learning_rate * (
            self.discount_factor * self.value(new_board, marker)
            - self.action_value[old_board, action])


def play_once_q_training(
    player_x: QPlayer,
    player_o: QPlayer,
    verbose = False,
    restrict_opening: bool = False,
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    >>> random.seed(1)
    >>> x, o = QPlayer(), QPlayer()
    >>> play_once_q_training(x, o)
    'X'
    >>> x.print_action_values()
    (('', '', ''), ('', '', ''), ('', '', '')): (2, 0, 'X') = 0.00000
    (('', '', ''), ('', '', ''), ('X', '', 'O')): (2, 1, 'X') = 0.00000
    (('', '', ''), ('', '', 'O'), ('X', 'X', 'O')): (0, 2, 'X') = 0.00000
    (('', '', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): (0, 1, 'X') = 0.00000
    (('O', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): (1, 1, 'X') = 1.00000
    >>> o.print_action_values()
    (('', '', ''), ('', '', ''), ('X', '', '')): (2, 2, 'O') = 0.00000
    (('', '', ''), ('', '', ''), ('X', 'X', 'O')): (1, 2, 'O') = 0.00000
    (('', '', 'X'), ('', '', 'O'), ('X', 'X', 'O')): (1, 0, 'O') = 0.00000
    (('', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): (0, 0, 'O') = -1.00000
    """
    previous_board: Board = ''
    previous_action: Action = ''
    board = get_init_board()
    players = {'X': player_x, 'O': player_o}
    score = 0
    game_over = False
    while not game_over:
        for marker, player in players.items():
            marker = cast(Marker, marker)
            other_player = players[get_other_marker(marker)]

            action = player.choose_action(board, marker, restrict_opening)
            new_board = get_updated_board(board, action)
            if verbose:
                print(new_board)

            game_over, score = is_game_over(new_board, marker)
            # Update the previous player's values based on this outcome.
            if previous_board:
                other_player.update_action_value(previous_board, previous_action, board, -score)
            # If it's game over, then this player too.
            if game_over:
                player.update_action_value(board, action, new_board, score)
                break
            previous_board, board = board, new_board
            previous_action = action

    if score > 0:
        return marker
    if score < 0:
        return get_other_marker(marker)
    return None
