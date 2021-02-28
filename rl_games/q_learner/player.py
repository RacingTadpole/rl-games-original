# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - Q training.
# Each player keeps a "Q table", ie. a mapping of (board, action) to values.
# The values are updated every turn using the Bellman equation.

import random
from dataclasses import dataclass, field
from typing import Generic, Tuple, Literal, Optional, Iterator, Dict, List, Sequence
from copy import deepcopy
from collections import defaultdict

from .game import State, Action, GameState


@dataclass
class Player(Generic[State, Action]):
    """
    """
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    action_value: Dict[Tuple[State, Action], float] = field(default_factory=lambda: defaultdict(float))
    discount_factor: float = 0.9

    def choose_action(self, game: GameState) -> Action:
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
        actions = list(game.get_actions())
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
                expected_reward = self.action_value.get((game.state, action), 0)
                if expected_reward > max_reward:
                    max_reward = expected_reward
                    best_action = action
            return best_action

    def value(self, game: GameState) -> float:
        """
        >>> random.seed(2)
        >>> player = Player(action_value={(1, 'a'): 2, (1, 'b'): 3, (1, 'c'): 7, (2, 'a'): 15})
        >>> player.value(1, None, get_actions=lambda _, __: ('a', 'b', 'c'))
        7
        """
        actions = list(game.get_actions())
        if len(actions):
            return max(self.action_value.get((game.state, action), 0) for action in actions)
        # If no actions are possible, the game must be over, and the value is 0.
        return 0

    def update_action_value(
        self,
        old_game_state: GameState,
        action: Action,
        new_game_state: GameState,
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
        self.action_value[old_game_state.state, action] += reward + self.learning_rate * (
            self.discount_factor * self.value(new_game_state)
            - self.action_value[old_game_state.state, action])



def play_multiplayer(
    game: GameState,
    players: Sequence[Player],
    verbose = False,
) -> Optional[Player]:
    """
    Returns the winner, if any.
    >>> random.seed(1)
    >>> x, o = QPlayer(), QPlayer()
    >>> play_multiplayer(NacGame, x, o)
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
