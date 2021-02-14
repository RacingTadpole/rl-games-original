# pylint: disable=unsubscriptable-object

# Noughts and crosses
# The game and player definitions.

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy

Marker = Literal['X', 'O']
Square = Optional[Marker]

size = 3

Row = Tuple[Square, Square, Square]
Board = Tuple[Row, Row, Row]

Action = Tuple[Literal[0, 1, 2], Literal[0, 1, 2]]

@dataclass
class Player:
    value: Dict[Board, float] = field(default_factory=dict)
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    base_value: float = 0


def get_actions(board: Board) -> Iterator[Action]:
    """
    >>> list(get_actions((('X', None, 'O'), ('X', 'O', 'O'), (None, None, 'X'))))
    [(0, 1), (2, 0), (2, 1)]
    """
    for r in range(size):
        for c in range(size):
            if board[r][c] is None:
                yield cast(Action, (r, c))

def get_updated_board(board: Board, action: Action, marker: Marker) -> Board:
    """
    >>> get_updated_board(get_init_board(), (2, 1), 'O')
    ((None, None, None), (None, None, None), (None, 'O', None))
    >>> get_updated_board((('X', None, 'O'), ('X', 'O', 'O'), (None, None, 'X')), (0, 1), 'X')
    (('X', 'X', 'O'), ('X', 'O', 'O'), (None, None, 'X'))
    """
    count = {'X': 0, 'O': 0}
    for r in range(size):
        for c in range(size):
            contents = board[r][c]
            if contents is not None:
                count[contents] += 1
    new_board = [list(row) for row in board]
    new_board[action[0]][action[1]] = marker
    return cast(Board, tuple(tuple(row) for row in new_board))

def choose_action(
    board: Board,
    player: Player,
    marker: Marker,
):
    """
    >>> board = (('X', 'X', 'O'), ('X', 'O', 'O'), (None, None, 'X'))
    >>> winning_board = (('X', 'X', 'O'), ('X', 'O', 'O'), ('O', None, 'X'))
    >>> value = { winning_board: 0.5 }
    >>> player = Player(value=value, explore_chance=0)
    >>> choose_action(board, player, 'O')
    (2, 0)
    """
    actions = list(get_actions(board))
    if random.uniform(0, 1) <= player.explore_chance:
        # Explore
        return random.choice(actions)
    else:
        # Greedy action - choose action with greatest expected value
        # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
        random.shuffle(actions)
        max_reward = -1
        best_action = actions[0]
        for action in actions:
            expected_reward = player.value.get(get_updated_board(board, action, marker), player.base_value)
            if expected_reward > max_reward:
                max_reward = expected_reward
                best_action = action
        return best_action

def get_init_board() -> Board:
    """
    >>> get_init_board()
    ((None, None, None), (None, None, None), (None, None, None))
    """
    return cast(Board, ((None,) * size,) * size)

def is_winner(b: Board, m: Marker) -> bool:
    """
    >>> is_winner((('X', 'X', 'O'), ('X', 'O', 'O'), (None, None, 'X')), 'O')
    False
    >>> is_winner((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', None, 'X')), 'O')
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
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), (None, None, 'X')), 'O')
    (False, 0)
    >>> is_game_over((('O', 'X', 'O'), ('X', 'O', 'X'), ('X', 'O', 'X')), 'X')
    (True, 0)
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', None, 'X')), 'O')
    (True, 1)
    >>> is_game_over((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', None, 'X')), 'X')
    (True, -1)
    """
    if is_winner(board, marker):
        return True, 1
    if is_winner(board, get_other_marker(marker)):
        return True, -1
    return all(cell for row in board for cell in row), 0

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

def play_once_no_training(
    player_x: Player,
    player_o: Player,
    verbose = False
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    No training.
    >>> random.seed(1)
    >>> x, o = Player(), Player()
    >>> [play_once_no_training(x, o) for _ in range(5)]
    ['X', 'O', None, 'X', 'X']
    """
    board = get_init_board()
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
            game_over, score = is_game_over(board, marker)
            if game_over:
                break

    marker = cast(Marker, marker)
    if score > 0:
        return marker
    if score < 0:
        return get_other_marker(marker)
    return None

def play_many(
    player_x: Player,
    player_o: Player,
    num_rounds = 1000,
    play_once = play_once_no_training,
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
        winner = play_once(player_x, player_o)
        if winner:
            count[winner] += 1
    return count['X'] / num_rounds, count['O'] / num_rounds
