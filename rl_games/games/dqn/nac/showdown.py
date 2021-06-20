# Run with:
#     python -m rl_games.games.dqn.nac.showdown

from typing import Sequence
from rl_games.games.nac import NacAction, NacState
from rl_games.q_learner.player import QPlayer
from rl_games.core.play import play_many
from rl_games.games.q_learner.nac_play import get_sample_game_and_trained_players as get_q
from rl_games.games.dqn.nac.play import get_sample_game_and_trained_players as get_dqn
from rl_games.games.dqn.nac2.play import get_sample_game_and_trained_players as get_dqn_2


def print_result(desc: str, result: dict) -> None:
    """
    >>> print_result('hello', {'O': 0.8})
    hello           0.00 0.80
    """
    print(f'{desc:15}', f"{result.get('X', 0):.2f}", f"{result.get('O', 0):.2f}")


def get_novice_players() -> Sequence[QPlayer]:
    return [
        QPlayer[NacState, NacAction]('X', explore_chance=1),
        QPlayer[NacState, NacAction]('O', explore_chance=1),
    ]


if __name__ == '__main__':
    game, players_q = get_q()
    _, players_dqn = get_dqn()
    _, players_dqn_2 = get_dqn_2()
    players_rand = get_novice_players()

    # The Q-trained player is the one to beat.
    print()
    print_result('rand v rand', play_many(game, [players_rand[0], players_rand[1]], range(100)))
    print()
    print_result('Q v Q', play_many(game, [players_q[0], players_q[1]], range(100)))
    print_result('rand v Q', play_many(game, [players_rand[0], players_q[1]], range(100)))
    print()
    print_result('Q v DQN', play_many(game, [players_q[0], players_dqn[1]], range(100)))
    print_result('DQN v Q', play_many(game, [players_dqn[0], players_q[1]], range(100)))
    print()
    print_result('DQN v rand', play_many(game, [players_dqn[0], players_rand[1]], range(100)))
    print_result('rand v DQN', play_many(game, [players_rand[0], players_dqn[1]], range(100)))
    print()
    print_result('Q v DQN2', play_many(game, [players_q[0], players_dqn_2[1]], range(100)))
    print_result('DQN2 v Q', play_many(game, [players_dqn_2[0], players_q[1]], range(100)))
    print()
    print_result('DQN2 v rand', play_many(game, [players_dqn_2[0], players_rand[1]], range(100)))
    print_result('rand v DQN2', play_many(game, [players_rand[0], players_dqn_2[1]], range(100)))
    print()
    print_result('DQN v DQN2', play_many(game, [players_dqn[0], players_dqn_2[1]], range(100)))
    print_result('DQN2 v DQN', play_many(game, [players_dqn_2[0], players_dqn[1]], range(100)))
