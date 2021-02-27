# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - Q training.
# Each player keeps a "Q table", ie. a mapping of (board, action) to values.
# The values are updated every turn using the Bellman equation.

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List
from copy import deepcopy
from collections import defaultdict

from .game import (
    Board, Player, Marker, Square, Action,
    x_marker, o_marker,
    get_init_board, get_updated_board, get_actions,
    get_other_marker, is_game_over, play_many
)


@dataclass
class QPlayer(Player):
    """
    Learn as X (first player) against a random opponent.
    >>> random.seed(2)
    >>> x, o = QPlayer(explore_chance=0.25), QPlayer(explore_chance=1)
    >>> a = play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    >>> x.explore_chance = 0
    >>> play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    (0.93, 0.031)

    Learn as O (second player) against a random opponent.
    >>> x, o = QPlayer(explore_chance=1), QPlayer(explore_chance=0.25)
    >>> a = play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    >>> o.explore_chance = 0
    >>> play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    (0.134, 0.73)

    Learn together.
    >>> x, o = QPlayer(explore_chance=0.25), QPlayer(explore_chance=0.25)
    >>> a = play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    >>> x.explore_chance = 0
    >>> o.explore_chance = 0
    >>> play_many(x, o, play_once=play_once_q_training, restrict_opening=True)
    (0.112, 0.057)
    """
    # Note the defaultdict defaults the action_value to 0, not to self.base_value.
    action_value: Dict[Tuple[Board, Action], float] = field(default_factory=lambda: defaultdict(float))
    discount_factor: float = 0.9

    def value(self, board: Board, marker: Marker, get_actions=get_actions) -> float:
        """
        >>> random.seed(2)
        >>> player = QPlayer(action_value={(1, 'a'): 2, (1, 'b'): 3, (1, 'c'): 7, (2, 'a'): 15})
        >>> player.value(1, None, get_actions=lambda _, __: ('a', 'b', 'c'))
        7
        """
        actions = list(get_actions(board, marker))
        if len(actions):
            return max(self.action_value.get((board, action), self.base_value)
                       for action in actions)
        # If no actions are possible, the game must be over, and the value is 0.
        return 0

    def choose_action(self, board: Board, marker: Marker, restrict_opening: bool = False) -> Action:
        """
        This differs from the simple version in that we use board directly, not the updated board.
        >>> board = (('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X'))
        >>> winning_board = (('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X'))
        >>> value = { winning_board: 0.5 }
        >>> @dataclass
        ... class DummyPlayer(Player):
        ...     def value(self, board: Board, marker: Marker) -> float:
        ...         return value.get(board, self.base_value)
        >>> player = DummyPlayer(explore_chance=0)
        >>> player.choose_action(board, 'O')
        (2, 0, 'O')
        """
        actions = list(get_actions(board, marker, restrict_opening))
        if random.uniform(0, 1) <= self.explore_chance:
            # Explore
            return random.choice(actions)
        else:
            # Greedy action - choose action with greatest expected value
            # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
            random.shuffle(actions)
            max_reward = -1.0
            best_action = actions[0]
            for action in actions:
                expected_reward = self.action_value.get((board, action), self.base_value)
                if expected_reward > max_reward:
                    max_reward = expected_reward
                    best_action = action
            return best_action

    def print_action_values(self):
        for (board, action), value in sorted(self.action_value.items()):
            print(f'{board}: {action} = {value:.5f}')

    def print_non_zero_action_values(self):
        for (board, action), value in sorted(self.action_value.items(), key=lambda item: (-item[1], item[0])):
            if value != 0:
                print(f'{board}: {action} = {value:.5f}')

    def print_values(self, print_zeroes = False):
        for board in sorted({k[0]: 1 for k in self.action_value.keys()}.keys()):
            x = self.value(board, x_marker)
            o = self.value(board, o_marker)
            if print_zeroes or x != 0 or o != 0:
                print(f'{board}: X={x:.5f} O={o:.5f}')

    def update_action_value(
        self,
        old_board: Board,
        action: Action,
        new_board: Board,
        reward: float,
    ) -> None:
        """
        Note that new_board is the next board in which this player can move again",
        ie. it includes an opponent's move.
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
) -> Square:
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
    >>> random.seed(1)
    >>> play_once_q_training(x, o)
    'O'
    >>> o.print_action_values()
    (('', '', ''), ('', '', ''), ('X', '', '')): (2, 2, 'O') = 0.00000
    (('', '', ''), ('', '', ''), ('X', 'X', 'O')): (1, 2, 'O') = 0.00000
    (('', '', 'X'), ('', '', 'O'), ('X', 'X', 'O')): (1, 0, 'O') = 0.00000
    (('', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): (0, 0, 'O') = -1.00000
    (('', 'X', 'X'), ('O', '', 'O'), ('X', 'X', 'O')): (1, 1, 'O') = 1.00000
    """
    previous_board: Optional[Board] = None
    previous_action: Optional[Action] = None
    board = get_init_board()
    players = {x_marker: player_x, o_marker: player_o}
    score = 0
    game_over = False
    while not game_over:
        for marker, player in players.items():
            other_player = players[get_other_marker(marker)]

            action = player.choose_action(board, marker, restrict_opening)
            new_board = get_updated_board(board, action)
            if verbose:
                print(new_board)

            game_over, score = is_game_over(new_board, marker)
            # Update the previous player's values based on this outcome.
            if previous_board and previous_action:
                if verbose:
                    print(f'updating previous player {previous_board} -- {previous_action}, {action} --> {new_board} = {-score}')
                # The update includes both players' moves.
                # From the previous player's point of view, this player's move is "part of the environment".
                other_player.update_action_value(previous_board, previous_action, new_board, -score)
            # If it's game over, then update this player too.
            if game_over:
                player.update_action_value(board, action, new_board, score)
                break
            previous_board, board = board, new_board
            previous_action = action

    if score > 0:
        return marker
    if score < 0:
        return get_other_marker(marker)
    return ''
