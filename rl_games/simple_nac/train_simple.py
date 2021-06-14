# Noughts and crosses
# Reinforcement Learning - simple training.
# This records the history of moves and updates the value function at
# all visited positions at the end of every game, starting at the end and
# looping back through the positions with:
#     prior  = player.value[state]
#     reward = prior + player.learning_rate * (reward - prior)
#     player.value[state] = reward

from dataclasses import dataclass, field
from typing import Dict, List

from .game import (
    Board, Player, Marker, Square,
    get_init_board, get_updated_board, get_other_marker, is_game_over,
    x_marker, o_marker, empty_square
)


@dataclass
class SimplePlayer(Player):
    """
    >>> import random
    >>> from .game import play_many
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
    ''
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

    def value(self, board: Board, _marker: Marker) -> float:
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
    verbose: bool = False,
    restrict_opening: bool = False,
) -> Square:
    """
    Returns the winner's marker, if any.
    >>> import random
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
    history: Dict[Marker, List[Board]] = {x_marker: [], o_marker: []}
    players = {x_marker: player_x, o_marker: player_o}
    score = 0
    game_over = False
    marker: Marker
    player: SimplePlayer
    while not game_over:
        for marker, player in players.items():
            action = player.choose_action(board, marker, restrict_opening)
            board = get_updated_board(board, action)
            if verbose:
                print(board)
            history[marker].append(board)
            game_over, score = is_game_over(board, marker)
            if game_over:
                break

    # Override the learning rate on the final board reward - just set it to the score. (Optional.)
    player._value[board] = score  # pylint: disable=protected-access
    player.update_values(history[marker][:-1], marker, score)

    other_marker = get_other_marker(marker)
    other_player = players[other_marker]
    other_player.update_values(history[other_marker], other_marker, -score)

    if score > 0:
        return marker
    if score < 0:
        return other_marker
    return empty_square
