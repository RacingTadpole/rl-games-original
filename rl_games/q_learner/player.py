# pylint: disable=unsubscriptable-object

# Noughts and crosses
# Reinforcement Learning - Q training.
# Each player keeps a "Q table", ie. a mapping of (board, action) to values.
# The values are updated every turn using the Bellman equation.
import sys
import random
from dataclasses import dataclass, field
from typing import Generic, Tuple, Literal, Optional, Iterator, Dict, List, Sequence
from collections import defaultdict

from .game import State, Action, Game


@dataclass
class Player(Generic[State, Action]):
    id: str = field(default_factory=lambda: f'{random.randrange(sys.maxsize)}')
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    action_value: Dict[Tuple[State, Action], float] = field(default_factory=lambda: defaultdict(float))
    discount_factor: float = 0.9

    def choose_action(self, game: Game[State, Action], state: State) -> Action:
        """
        >>> from .countdown import Countdown
        >>> random.seed(3)
        >>> game = Countdown()
        >>> player = Player[int, int]('A', explore_chance=0)
        >>> player.choose_action(game, 5), player.choose_action(game, 5)
        (2, 3)
        """
        actions = list(game.get_actions(state))
        if random.uniform(0, 1) <= self.explore_chance:
            # Explore
            return random.choice(actions)
        else:
            # Greedy action - choose action with greatest expected value
            # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
            random.shuffle(actions)
            max_reward = -1.0
            if len(actions) == 0:
                raise IndexError(f'No actions available from {state}')
            best_action = actions[0]
            for action in actions:
                expected_reward = self.action_value.get((state, action), 0)
                if expected_reward > max_reward:
                    max_reward = expected_reward
                    best_action = action
            return best_action

    def value(self, game: Game, state: State) -> float:
        """
        >>> from .countdown import Countdown
        >>> random.seed(2)
        >>> game = Countdown()
        >>> player = Player[int, int](action_value={(1, 1): 1, (1, 2): 0, (2, 3): -7, (3, 3): 2})
        >>> player.value(game, 1), player.value(game, 2), player.value(game, 3)
        (1, 0, 2)
        """
        actions = list(game.get_actions(state))
        if len(actions):
            return max(self.action_value.get((state, action), 0) for action in actions)
        # If no actions are possible, the game must be over, and the value is 0.
        return 0

    def update_action_value(
        self,
        game: Game,
        old_state: State,
        action: Action,
        new_state: State,
        reward: float,
    ) -> None:
        """
        Note that new_state is the next state in which this player can move again,
        ie. it includes opponent moves.
        We'll imagine the dummy game is a 2-player game.
        >>> from .countdown import Countdown
        >>> random.seed(2)
        >>> game = Countdown()
        >>> player = Player()
        >>> player.update_action_value(game, 4, True, 6, 1)
        >>> player.update_action_value(game, 2, True, 4, 0)
        >>> player.update_action_value(game, 0, True, 2, 0)
        >>> {k: float(f'{v:.4f}') for k, v in player.action_value.items()}
        {(4, True): 1.0, (2, True): 0.09, (0, True): 0.0081}
        """
        self.action_value[old_state, action] += reward + self.learning_rate * (
            self.discount_factor * self.value(game, new_state)
            - self.action_value[old_state, action])
