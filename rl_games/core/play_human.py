from typing import Union, Sequence, Optional
from rl_games.q_learner.player import QPlayer
from .player import Player
from .play import play_many
from .game import Game, State, Action

def get_human_action(game: Game[State, Action], state: State, player_name: str) -> Action:
    actions = list(game.get_actions(state))
    choice = 0
    while choice < 1 or choice > len(actions):
        print(f'Your turn {player_name}. You can choose:')
        for index, action in enumerate(actions):
            print(f'{index + 1}.', action)
        choice_str = input('Please choose a number: ')
        try:
            choice = int(choice_str)
        except:
            choice = 1
    return actions[choice - 1]


def play_human(
    game: Game[State, Action],
    players: Sequence[Union[Player[State, Action], str]],
    verbose: bool = False,
) -> Optional[Union[Player[State, Action], str]]:
    """
    Plays a multiplayer game against a human to the end, and reports the winner.
    Pass a string representing the name of the player for any human players.
    You should preset the players to have no chance of exploring.
    Does not further train the players.
    """
    state = game.get_init_state()
    game_over = False
    while not game_over:
        for player in players:
            print()
            print(state)
            print()
            if isinstance(player, str):
                # Ask the human for their action.
                action = get_human_action(game, state, player)
            else:
                # Choose an AI action.
                action = player.choose_action(game, state)
                print(f'{player.id}: {action}')
            new_state = game.updated(state, action)
            reward, game_over = game.get_score_and_game_over(new_state)
            # Finally, update the state for the next player.
            state = new_state
            if game_over:
                print()
                print(new_state)
                print('Game over!')
                print()
                break

    if reward > 0:
        return player
    if reward < 0:
        # If there are two players, the other player must have won.
        # Otherwise, there was no winner, only a loser.
        if len(players) == 2:
            return players[0] if player == players[1] else players[1]
    return None


def play_human_ui(game: Game[State, Action], trained_players: Sequence[Player[State, Action]]) -> None:
    """
    Play an interactive game with the human.
    """
    while True:
        try:
            index_str = input(f'Play as player number {tuple(range(1, 1 + len(trained_players)))}? ')
            if index_str == '':
                break
            index = int(index_str)
        except ValueError:
            index = 1
        new_players: list[Union[Player[State, Action], str]] = ['human' if i == index - 1 else p for i, p in enumerate(trained_players)]
        winner = play_human(game, new_players)

        if winner:
            if isinstance(winner, str):
                print(f'Congratulations, the winner was {winner}!')
            else:
                print(f'The winner was {winner.id}')
        else:
            print('The game was a draw.')
        print()
