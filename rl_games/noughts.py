# Reinforcement Learning
# Noughts and crosses

import random
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy

# pylint: disable=unsubscriptable-object
Marker = Literal['X', 'O']
Square = Optional[Marker]

size = 3

Row = Tuple[Square, Square, Square]
Board = Tuple[Row, Row, Row]

Action = Tuple[Literal[0, 1, 2], Literal[0, 1, 2]]

def get_actions(board: Board) -> Iterator[Action]:
    """
    >>> list(get_actions((('X', None, 'O'), ('X', 'O', 'O'), (None, None, 'X'))))
    [(0, 1), (2, 0), (2, 1)]
    """
    for r in range(size):
        for c in range(size):
            if board[r][c] is None:
                yield cast(Action, (r, c))

def get_updated_board(board: Board, action: Action, player: Marker) -> Board:
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
    new_board[action[0]][action[1]] = player
    return tuple(tuple(row) for row in new_board)

def choose_action(
    board: Board,
    value: Dict[Board, float],
    player: Marker,
    explore_chance = 0.1,
    base_value = 0
):
    """
    >>> board = (('X', 'X', 'O'), ('X', 'O', 'O'), (None, None, 'X'))
    >>> winning_board = (('X', 'X', 'O'), ('X', 'O', 'O'), ('O', None, 'X'))
    >>> value = { winning_board: 0.5 } 
    >>> choose_action(board, value, 'O', explore_chance=0)
    (2, 0)
    """
    actions = list(get_actions(board))
    if random.uniform(0, 1) <= explore_chance:
        # Explore
        return random.choice(actions)
    else:
        # Greedy action - choose action with greatest expected value
        # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
        random.shuffle(actions)
        expected_rewards = tuple((value.get(get_updated_board(board, action, player), base_value), action)
                                 for action in actions)
        max_expected_reward = max(expected_rewards)
        return max_expected_reward[1]

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

def get_other_player(player: Marker) -> Marker:
    """
    >>> get_other_player('X')
    'O'
    >>> get_other_player('O')
    'X'
    """
    return 'O' if player == 'X' else 'X'

def is_game_over(board: Board, player: Marker) -> Tuple[bool, int]:
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
    if is_winner(board, player):
        return True, 1
    if is_winner(board, get_other_player(player)):
        return True, -1
    return all(cell for row in board for cell in row), 0

def update_values(
    states: List[Board],
    value: Dict[Board, float],
    final_reward: float,
    learning_rate: float,
    default_value = 0
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
    >>> value = {}
    >>> update_values(states, value, 1, 0.1)
    >>> for k,v in value.items():
    ...     print(f'{k}: {v}')
    (('X', None, 'O'), ('O', 'O', None), ('X', None, 'X')): 0.1
    (('X', None, 'O'), (None, 'O', None), (None, None, 'X')): 0.01
    (('X', None, None), (None, 'O', None), (None, None, 'X')): 0.001
    (('X', None, None), (None, None, None), (None, None, None)): 0.0001
    """
    reward = final_reward
    for state in states[-1::-1]:
        prior = value.get(state, default_value)
        reward = prior + learning_rate * (reward - prior)
        value[state] = round(reward, 5)

def play_once(
    player_value: Dict[Marker, Dict[Board, float]] = None,
    learning_rate = 0.1,
    explore_chance = 0.1,
    verbose = False
) -> Dict[Marker, Dict[Board, float]]:
    """
    >>> random.seed(3)
    >>> player_value = play_once()
    >>> for player in ('X', 'O'):
    ...     for k, v in player_value[player].items():
    ...         print(f'{player}: {k}: {v}')
    X: ((None, None, 'X'), ('O', 'X', 'O'), ('X', 'O', 'X')): 1
    X: ((None, None, None), (None, 'X', 'O'), ('X', 'O', 'X')): 0.1
    X: ((None, None, None), (None, None, None), ('X', 'O', 'X')): 0.01
    X: ((None, None, None), (None, None, None), (None, None, 'X')): 0.001
    O: ((None, None, None), ('O', 'X', 'O'), ('X', 'O', 'X')): -0.1
    O: ((None, None, None), (None, None, 'O'), ('X', 'O', 'X')): -0.01
    O: ((None, None, None), (None, None, None), (None, 'O', 'X')): -0.001
    """
    if player_value is None:
        player_value = {'X': {}, 'O': {}}
    board = get_init_board()
    history: Dict[Marker, List[Board]] = {'X': [], 'O': []}
    score = 0
    game_over = False
    last_player = None
    while not game_over:
        for player in cast(Tuple[Marker], ('X', 'O')):
            action = choose_action(board, player_value[player], player, explore_chance=explore_chance)
            board = get_updated_board(board, action, player)
            if verbose:
                print(board)
            history[player].append(board)
            game_over, score = is_game_over(board, player)
            if game_over:
                last_player = player
                break

    # Override the learning rate on the final board reward - just set it to the score. (Optional.)
    player_value[last_player][board] = score
    update_values(history[last_player][:-1], player_value[last_player], score, learning_rate)

    other = get_other_player(last_player)
    update_values(history[other], player_value[other], -score, learning_rate)

    return player_value

def play_many(
    num_rounds = 1000,
    learning_rate = 0.1,
    explore_chance = 0.1
) -> Dict[Marker, Dict[Board, float]]:
    """
    >>> random.seed(3)
    >>> player_value = play_many()
    >>> v = play_once(player_value, verbose=True)
    ((None, None, None), (None, None, None), (None, None, 'X'))
    ((None, None, None), (None, 'O', None), (None, None, 'X'))
    ((None, None, 'X'), (None, 'O', None), (None, None, 'X'))
    ((None, None, 'X'), (None, 'O', 'O'), (None, None, 'X'))
    ((None, None, 'X'), ('X', 'O', 'O'), (None, None, 'X'))
    ((None, None, 'X'), ('X', 'O', 'O'), ('O', None, 'X'))
    ((None, None, 'X'), ('X', 'O', 'O'), ('O', 'X', 'X'))
    ((None, 'O', 'X'), ('X', 'O', 'O'), ('O', 'X', 'X'))
    (('X', 'O', 'X'), ('X', 'O', 'O'), ('O', 'X', 'X'))
    """
    player_value = {'X': {}, 'O': {}}
    for _ in range(num_rounds):
        player_value = play_once(player_value, learning_rate, explore_chance)
    return player_value
