# Reinforcement Learning
# Noughts and crosses

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, cast
from copy import deepcopy

# pylint: disable=unsubscriptable-object
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

def play_once(
    player_x: Player,
    player_o: Player,
    verbose = False
) -> Optional[Marker]:
    """
    Returns the winner's marker, if any.
    >>> random.seed(1)
    >>> x, o = Player(), Player()
    >>> play_once(x, o)
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

def play_many(
    player_x: Player,
    player_o: Player,
    num_rounds = 1000,
) -> Tuple[float, float]:
    """
    Returns the fraction won by player x and o.
    >>> random.seed(2)
    >>> x, o = Player(), Player()
    >>> play_many(x, o)
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
    >>> play_many(x, o)
    (0.166, 0.09)

    An experienced player against a novice should win
    >>> random.seed(2)
    >>> play_many(x, Player())
    (0.288, 0.112)
    >>> play_many(x, Player())
    (0.207, 0.01)

    Similarly for the player playing O, although this takes more training
    >>> random.seed(2)
    >>> o.explore_chance = 0.1
    >>> play_many(Player(explore_chance=0.5), o)
    (0.363, 0.43)
    >>> o.explore_chance = 0
    >>> play_many(Player(), o, 100)
    (0.17, 0.44)
    """
    count = {'X': 0, 'O': 0}
    for _ in range(num_rounds):
        winner = play_once(player_x, player_o)
        if winner:
            count[winner] += 1
    return count['X'] / num_rounds, count['O'] / num_rounds
