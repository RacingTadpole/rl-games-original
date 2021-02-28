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

from .game import State, Action, Game


@dataclass
class Player(Generic[State, Action]):
    """
    """
    learning_rate: float = 0.1
    explore_chance: float = 0.1
    action_value: Dict[Tuple[State, Action], float] = field(default_factory=lambda: defaultdict(float))
    discount_factor: float = 0.9

    def choose_action(self, game: Game, state: State) -> Action:
        actions = list(game.get_actions(state))
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
                expected_reward = self.action_value.get((state, action), 0)
                if expected_reward > max_reward:
                    max_reward = expected_reward
                    best_action = action
            return best_action

    def value(self, game: Game, state: State) -> float:
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
        self.action_value[old_state, action] += reward + self.learning_rate * (
            self.discount_factor * self.value(game, new_state)
            - self.action_value[old_state, action])


@dataclass
class TurnRecord(Generic[State, Action]):
    """
    A TurnRecord keeps track across other players' turns, what the player saw and did,
    and then what the total reward has been to them since.
    """
    state: Optional[State] = None
    action: Optional[Action] = None
    reward: float = 0


def play_multiplayer(
    game: Game,
    players: Sequence[Player],
    verbose = False,
) -> Optional[Player]:
    turn_records: List[TurnRecord] = [TurnRecord()] * len(players)
    state = game.get_init_state()
    score = 0
    game_over = False
    while not game_over:
        for index, player in enumerate(players):
            turn_record = turn_records[index]
            # First update based on the outcome since your last turn.
            if turn_record.state and turn_record.action:
                if verbose:
                    print(f'updating player {turn_record.state} -- {turn_record.action} --> {state} = {turn_record.reward}')
                player.update_action_value(game, turn_record.state, turn_record.action, state, turn_record.reward)

            # Now have your turn.
            action = player.choose_action(game, state)
            new_state = game.updated(state, action)
            reward, game_over = game.get_score_and_game_over(new_state)
            # Update this player's turn record with the state they were presented with, the chosen action, and the reward so far.
            turn_records[index] = TurnRecord(state, action, reward)
            # Update the previous players' rewards - we assume they get -reward.
            for j in range(len(players)):
                if j != index:
                    turn_records[j].reward -= reward
            # Finally, update the state for the next player.
            state = new_state
            if game_over:
                break

    # When the game ends, update all players.
    for j, p in enumerate(players):
        turn_record = turn_records[j]
        p.update_action_value(game, turn_record.state, turn_record.action, state, turn_record.reward)
    if reward > 0:
        return p
    if score < 0:
        raise NotImplementedError('No current way to tell the winner if the final score is negative.')
    return None
