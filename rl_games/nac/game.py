# pylint: disable=unsubscriptable-object

# Noughts and crosses
# The game and player definitions.

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, Sequence, cast
from copy import deepcopy

Marker = Literal['X', 'O']
Square = Optional[Marker]

size = 3

Row = Tuple[Square, Square, Square]
Board = Tuple[Row, Row, Row]

Action = Tuple[Literal[0, 1, 2], Literal[0, 1, 2], Marker]

def get_actions(board: Board, marker: Marker, restrict_opening: bool = False) -> Iterator[Action]:
    """
    >>> list(get_actions((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')), 'X'))
    [(0, 1, 'X'), (2, 0, 'X'), (2, 1, 'X')]
    >>> list(get_actions((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')), 'X'))
    []
    >>> len(get_actions((('', '', ''), ('', '', ''), ('', '', '')), 'X'))
    9
    >>> list(get_actions((('', '', ''), ('', '', ''), ('', '', '')), 'X', restrict_opening=True))
    [(0, 0, 'X'), (1, 0, 'X'), (1, 1, 'X')]
    """
    if marker in get_valid_next_markers(board):
        if restrict_opening and board == get_init_board():
            for r, c in [(0, 0), (1, 0), (1, 1)]:
                yield cast(Action, (r, c, marker))
        else:
            for r in range(size):
                for c in range(size):
                    if board[r][c] == '':
                        yield cast(Action, (r, c, marker))

@dataclass
class Player:
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    base_value: float = 0

    def value(self, board: Board, marker: Marker) -> float:
        # Subclass Player to improve this.
        return self.base_value

    def choose_action(self, board: Board, marker: Marker, restrict_opening: bool = False) -> Action:
        """
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
            max_reward = -1
            best_action = actions[0]
            for action in actions:
                expected_reward = self.value(get_updated_board(board, action), marker)
                if expected_reward > max_reward:
                    max_reward = expected_reward
                    best_action = action
            return best_action


def get_valid_next_markers(board: Board) -> Sequence[Marker]:
    """
    >>> get_valid_next_markers((('', '', ''), ('', '', ''), ('', 'O', '')))
    ('X',)
    >>> get_valid_next_markers((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
    ('X', 'O')
    >>> get_valid_next_markers((('X', 'O', 'O'), ('', 'O', 'O'), ('', '', 'X')))
    ()
    """
    count = {'X': 0, 'O': 0}
    for r in range(size):
        for c in range(size):
            contents = board[r][c]
            if contents != '':
                count[contents] += 1
    if count['X'] == count['O'] + 1:
        return ('O',)
    if count['X'] == count['O'] - 1:
        return ('X',)
    if count['X'] == count['O']:
        return ('X', 'O')
    return ()

def get_updated_board(board: Board, action: Action) -> Board:
    """
    >>> get_updated_board(get_init_board(), (2, 1, 'O'))
    (('', '', ''), ('', '', ''), ('', 'O', ''))
    >>> get_updated_board((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')), (0, 1, 'X'))
    (('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X'))
    """
    new_board = [list(row) for row in board]
    marker = action[-1]
    new_board[action[0]][action[1]] = marker
    return cast(Board, tuple(tuple(row) for row in new_board))

def get_init_board() -> Board:
    """
    >>> get_init_board()
    (('', '', ''), ('', '', ''), ('', '', ''))
    """
    return cast(Board, (('',) * size,) * size)

def is_winner(b: Board, m: Marker) -> bool:
    """
    >>> is_winner((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')), 'O')
    False
    >>> is_winner((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')), 'O')
    True
    """
    return \
        any(all(b[r][c] == m for c in range(size)) for r in range(size)) or \
        any(all(b[r][c] == m for r in range(size)) for c in range(size)) or \
        all(b[d][d] == m for d in range(size)) or \
        all(b[d][size - 1 - d] == m for d in range(size))

def get_other_marker(marker: Marker) -> Marker:
    """
    >>> get_other_marker('X')
    'O'
    >>> get_other_marker('O')
    'X'
    """
    return 'O' if marker == 'X' else 'X'

def is_game_over(board: Board, marker: Marker) -> Tuple[bool, int]:
    """
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')), 'O')
    (False, 0)
    >>> is_game_over((('O', 'X', 'O'), ('X', 'O', 'X'), ('X', 'O', 'X')), 'X')
    (True, 0)
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')), 'O')
    (True, 1)
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')), 'X')
    (True, -1)
    """
    if is_winner(board, marker):
        return True, 1
    if is_winner(board, get_other_marker(marker)):
        return True, -1
    return all(cell for row in board for cell in row), 0

def play_once_no_training(
    player_x: Player,
    player_o: Player,
    restrict_opening: bool = False,
    verbose = False
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    No training.
    >>> random.seed(1)
    >>> x, o = Player(), Player()
    >>> [play_once_no_training(x, o) for _ in range(5)]
    ['X', 'O', '', 'X', 'X']
    """
    board = get_init_board()
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
            game_over, score = is_game_over(board, marker)
            if game_over:
                break

    marker = cast(Marker, marker)
    if score > 0:
        return marker
    if score < 0:
        return get_other_marker(marker)
    return ''

def play_many(
    player_x: Player,
    player_o: Player,
    num_rounds = 1000,
    play_once = play_once_no_training,
    restrict_opening: bool = False,
) -> Tuple[float, float]:
    """
    Returns the fraction won by player x and o.
    These are untrained players.
    >>> random.seed(2)
    >>> x, o = Player(), Player()
    >>> play_many(x, o)
    (0.618, 0.264)
    """
    count = {'X': 0, 'O': 0}
    for _ in range(num_rounds):
        winner = play_once(player_x, player_o, restrict_opening=restrict_opening)
        if winner:
            count[winner] += 1
    return count['X'] / num_rounds, count['O'] / num_rounds
