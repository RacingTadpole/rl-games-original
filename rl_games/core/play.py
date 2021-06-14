from dataclasses import dataclass
from typing import Generic, Optional, Dict, List, Sequence
from collections import defaultdict

from .game import State, Action, Game
from .player import Player


@dataclass
class TurnRecord(Generic[State, Action]):
    """
    A TurnRecord keeps track across other players' turns, what the player saw and did,
    and then what the total reward has been to them since.
    """
    state: Optional[State] = None
    action: Optional[Action] = None
    reward: float = 0


def play(
    game: Game,
    players: Sequence[Player],
    verbose: bool = False,
) -> Optional[Player]:
    """
    Plays a multiplayer game to the end, and reports the winner.
    Updates each player.

    >>> import random
    >>> from rl_games.games.countdown import Countdown
    >>> from rl_games.q_learner.player import QPlayer
    >>> game = Countdown()
    >>> def nice_action_value(player: Player):
    ...    return player.id, {k: float(f'{v:.4f}') for k,v in player.action_value.items() if v != 0}

    1 player
    >>> players = [QPlayer('A')]
    >>> random.seed(2)
    >>> nice_action_value(play(game, players))
    ('A', {(2, 2): 1.0})
    >>> nice_action_value(play(game, players))
    ('A', {(2, 2): 1.0, (3, 3): 1.0})
    >>> nice_action_value(play(game, players))
    ('A', {(2, 2): 1.9, (3, 3): 1.0, (5, 3): 0.09})

    3 players
    >>> players = QPlayer('A'), QPlayer('B'), QPlayer('C')
    >>> random.seed(2)
    >>> play(game, players).id
    'C'
    >>> [nice_action_value(p) for p in players]
    [('A', {(6, 3): -1.0}), ('B', {(3, 1): -1.0}), ('C', {(2, 2): 1.0})]
    >>> play(game, players).id
    'C'
    >>> [nice_action_value(p) for p in players]
    [('A', {(6, 3): -1.0, (7, 3): -1.0}), ('B', {(3, 1): -1.0, (4, 1): -1.0}), ('C', {(2, 2): 1.0, (3, 3): 1.0})]
    """
    turn_records: List[TurnRecord] = [TurnRecord()] * len(players)
    state = game.get_init_state()
    game_over = False
    player: Optional[Player] = None
    while not game_over:
        for index, player in enumerate(players):
            turn_record = turn_records[index]
            # First update based on the outcome since your last turn.
            if turn_record.state is not None and turn_record.action is not None:
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
        return player
    if reward < 0:
        # If there are two players, the other player must have won.
        # Otherwise, there was no winner, only a loser.
        if len(players) == 2:
            return players[0] if player == players[1] else players[1]
    return None

def play_many(
    game: Game,
    players: Sequence[Player],
    num_rounds: int = 1000,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Returns the fraction won by each player.
    Starting at 20, B can always win.

    >>> import random
    >>> from rl_games.games.countdown import Countdown
    >>> from rl_games.q_learner.player import QPlayer
    >>> game = Countdown(start=20)
    >>> random.seed(2)
    >>> a, b = QPlayer('A'), QPlayer('B')
    >>> play_many(game, [a, b])
    {'A': 0.243, 'B': 0.757}

    If we had started at 21, then A can always win.
    >>> game = Countdown(start=21)
    >>> random.seed(2)
    >>> a, b = QPlayer('A'), QPlayer('B')
    >>> play_many(game, [a, b])
    {'A': 0.693, 'B': 0.307}
    """
    count: Dict[str, float] = defaultdict(float)
    for _ in range(num_rounds):
        winner = play(game, players, verbose=verbose)
        if winner:
            count[winner.id] += 1
    return {player_id: total / num_rounds for player_id, total in count.items()}
