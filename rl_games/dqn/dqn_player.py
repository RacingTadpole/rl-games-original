# Reinforcement Learning - deep q network.
# Each player uses a neural network model.

# To start with, let's just implement a noughts-and-crosses player here.
# The state is:  board: Tuple[Tuple[Square, ...], ...]; next_player_index: PlayerIndex = 0
# For simplicitly let's encode the board as size^2 squares, each of which could be any of 3 states,
# and the next turn as either player, ie. (size * size * 3) * 2.
# A NAC action is just a position on the board, ie. size * size.
# For simplicity let's assume size = 3, ie. 54 states and 9 actions.

import random
from dataclasses import dataclass
from typing import Generic, Any
import numpy as np

from rl_games.core.game import State, Action, Game
from rl_games.core.player import Player
from rl_games.neural.neural_network import NeuralNetwork
from .setup import DqnSetup


@dataclass
class DqnPlayer(Player, Generic[State, Action]):
    dqn: DqnSetup
    explore_chance: float = 0.1
    discount_factor: float = 0.9

    def __post_init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.model = NeuralNetwork(
            input_size=self.dqn.num_states,
            hidden_size=self.dqn.hidden_size,
            output_size=self.dqn.num_actions
        )

    def choose_action(self, game: Game[State, Action], state: State) -> Action:
        """
        A player is initialized with a randomly weighted model, so choose a first action for 5 different players.
        >>> from rl_games.games.nac import Nac, NacState, NacAction
        >>> from rl_games.games.dqn.nac.setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> [DqnPlayer('A', NacDqnSetup(), explore_chance=0).choose_action(game, game.get_init_state()) for _ in range(5)]
        [(1, 2), (1, 0), (2, 0), (0, 2), (2, 2)]
        """
        action_mask = self.dqn.get_action_mask(game, state)
        if np.all(action_mask):
            raise IndexError(f'No actions available from {state}')
        if random.uniform(0, 1) <= self.explore_chance:
            # Explore
            actions = list(game.get_actions(state))
            return random.choice(actions)
        # Greedy action - choose action with greatest expected value
        # Shuffle the actions (in place) to randomly choose between top-ranked equal-valued rewards
        model_input = self.dqn.get_input_vector(game, state)
        model_output = self.model.predict(model_input)
        action, _ = self.dqn.get_action_and_value_from_output(game, model_output, action_mask)
        return action  # type: ignore

    def value(self, game: Game, state: State) -> float:
        """
        >>> from rl_games.games.nac import Nac, NacState, NacAction
        >>> from rl_games.games.dqn.nac.setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> player = DqnPlayer[NacState, NacAction]('A', NacDqnSetup())
        >>> round(player.value(game, game.get_init_state()), 4)
        0.0214
        """
        action_mask = self.dqn.get_action_mask(game, state)
        if not np.all(action_mask):
            model_output = self.model.predict(self.dqn.get_input_vector(game, state))
            return self.dqn.get_action_and_value_from_output(game, model_output, action_mask)[1]
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
        >>> from rl_games.games.dqn.nac.setup import NacDqnSetup
        >>> random.seed(3); np.random.seed(3)
        >>> game = Nac()
        >>> player = DqnPlayer('X', NacDqnSetup())

        Set up this game: [..X/.../...] -> [..X/.O./...] -> [.XX/.O./...] -> [.XX/.O./..O] -> [XXX/.O./..O]
        After this game, the likelihood of X making the final winning move in position (0, 0) should have increased more than any other changes.

        >>> states = [game.get_init_state()]
        >>> actions = [NacAction(row=0, col=2), NacAction(row=1, col=1), NacAction(row=0, col=1), NacAction(row=2, col=2), NacAction(row=0, col=0)]
        >>> for action in actions:
        ...     states.append(game.updated(states[-1], action))

        >>> x = player.model.predict(player.dqn.get_input_vector(game, states[4]))
        >>> [round(a, 2) for a in x.flatten().tolist()]
        [-0.01, 0.02, -0.01, -0.05, -0.02, 0.02, -0.0, 0.01, 0.01]

        >>> player.update_action_value(game, states[0], actions[0], states[2], 0)
        >>> player.update_action_value(game, states[2], actions[2], states[4], 0)
        >>> player.update_action_value(game, states[4], actions[4], states[5], 1)

        >>> x = player.model.predict(player.dqn.get_input_vector(game, states[4]))
        >>> [round(a, 2) for a in x.flatten().tolist()]
        [0.11, 0.02, -0.0, -0.05, -0.02, 0.02, -0.0, 0.01, 0.01]
        """
        target = reward + self.discount_factor * np.max(
            self.model.predict(self.dqn.get_input_vector(game, new_state)))
        target_vector = self.model.predict(self.dqn.get_input_vector(game, old_state))
        target_vector[0][self.dqn.get_onehot_index_from_action(game, action)] = target
        self.model.train([self.dqn.get_input_vector(game, old_state)], [target_vector], num_iterations=1)
