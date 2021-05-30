# pylint: disable=unsubscriptable-object

# Reinforcement Learning - deep q network.
# Each player uses a neural network model.

# To start with, let's just implement a noughts-and-crosses player here.
# The state is:  board: Tuple[Tuple[Square, ...], ...]; next_turn: Marker = x_marker
# For simplicitly let's encode the board as size^2 squares, each of which could be any of 3 states,
# and the next turn as either player, ie. (size * size * 3) * 2.
# A NAC action is just a position on the board, ie. size * size.
# For simplicity let's assume size = 3, ie. 54 states and 9 actions.

import sys
import random
import numpy as np
from dataclasses import dataclass
from typing import Generic, Tuple, Literal, Optional, Iterator, Dict, List, Sequence, Any, Callable
from collections import defaultdict

from rl_games.core.game import State, Action, Game
from rl_games.core.player import Player
from rl_games.games.nac import Nac, NacState, NacAction, x_marker, o_marker, empty_square
from rl_games.neural.neural_network import NeuralNetwork
from .nac_setup import DqnSetup


@dataclass
class DqnPlayer(Player, Generic[State, Action]):
    dqn: DqnSetup
    explore_chance: float = 0.1
    discount_factor: float = 0.9

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.model = NeuralNetwork(
            input_size=self.dqn.num_states,
            hidden_size=self.dqn.hidden_size,
            output_size=self.dqn.num_actions
        )

    def choose_action(self, game: Game[State, Action], state: State) -> Action:
        """
        A player is initialized with a randomly weighted model, so choose a first action for 5 different players.
        >>> from rl_games.games.nac import Nac, NacState, NacAction
        >>> from .nac_setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> [DqnPlayer('A', NacDqnSetup(), explore_chance=0).choose_action(game, game.get_init_state()) for _ in range(5)]
        [(0, 2), (0, 0), (2, 2), (0, 1), (1, 2)]
        """
        actions = list(game.get_actions(state))
        if random.uniform(0, 1) <= self.explore_chance:
            # Explore
            return random.choice(actions)
        # Greedy action - choose action with greatest expected value
        # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
        random.shuffle(actions)
        if len(actions) == 0:
            raise IndexError(f'No actions available from {state}')
        model_input = self.dqn.get_input_vector(state)
        model_output = self.model.predict(model_input)
        action, _ = self.dqn.get_action_and_value_from_output(game, model_output, actions)
        return action  # type: ignore

    def value(self, game: Game, state: State) -> float:
        """
        >>> from rl_games.games.nac import Nac, NacState, NacAction
        >>> from .nac_setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> player = DqnPlayer[NacState, NacAction]('A', NacDqnSetup())
        >>> f'{player.value(game, game.get_init_state()):.4}'
        '2.742'
        """
        actions = list(game.get_actions(state))
        if len(actions):
            model_output = self.model.predict(self.dqn.get_input_vector(state))
            return self.dqn.get_action_and_value_from_output(game, model_output, actions)[1]
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
        >>> from rl_games.games.nac import Nac, NacState, NacAction
        >>> from .nac_setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> player = DqnPlayer('A', NacDqnSetup())
        >>> empty = game.get_init_state()
        >>> player.update_action_value(game, 4, True, 6, 1)
        >>> player.update_action_value(game, 2, True, 4, 0)
        >>> player.update_action_value(game, empty, True, 2, 0)
        """
        target = reward + self.discount_factor * np.max(
            self.model.predict(self.dqn.get_input_vector(new_state)))
        target_vector = self.model.predict(self.dqn.get_input_vector(old_state))
        target_vector[0][self.dqn.get_onehot_index_from_action(game, action)] = target
        self.model.train([self.dqn.get_input_vector(old_state)], [target_vector], num_iterations=1)
