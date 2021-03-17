# pylint: disable=unsubscriptable-object

# Run with:
#     python -m rl_games.q_learner.chopsticks_play_human

if __name__ == '__main__':
    from typing import Union
    from rl_games.q_learner.player import Player
    from rl_games.core.play import play_human, play_many
    from rl_games.games.chopsticks import Chopsticks, ChopsticksState, ChopsticksAction

    players: list[Player[ChopsticksState, ChopsticksAction]] = [Player('P1'), Player('P2')]
    game = Chopsticks()

    play_many(game, players, 25000)
    for player in players:
        player.explore_chance = 0.1
    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0.05
    play_many(game, players, 9000)
    for player in players:
        player.explore_chance = 0

    while True:
        try:
            index_str = input(f'Play as player number {tuple(range(1, 1 + len(players)))}? ')
            if index_str == '':
                break
            index = int(index_str)
        except ValueError:
            index = 1
        new_players: list[Union[Player[ChopsticksState, ChopsticksAction], str]] = ['human' if i == index - 1 else p for i, p in enumerate(players)]
        winner = play_human(game, new_players)

        if winner is not None:
            if isinstance(winner, str):
                print(f'Congratulations, the winner was {winner}!')
            else:
                print(f'The winner was {winner.id}')
        else:
            print('The game was a draw.')
        print()
