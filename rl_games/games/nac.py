# Noughts and crosses

from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, cast, Generator
from rl_games.core.game import Game, PlayerIndex

Marker = Literal['X', 'O']
Square = Literal['X', 'O', '']

x_marker, o_marker, empty_square = cast(Marker, 'X'), cast(Marker, 'O'), cast(Square, '')

@dataclass(frozen=True, repr=False)
class NacAction:
    row: int
    col: int

    def __repr__(self) -> str:
        return f'({self.row}, {self.col})'

    def __str__(self) -> str:
        """
        >>> print(NacAction(1, 2))
        B3
        """
        return f'{chr(ord("A") + self.row)}{self.col + 1}'


@dataclass(frozen=True)
class NacState:
    board: Tuple[Tuple[Square, ...], ...] = ()
    next_player_index: PlayerIndex = 0

    def __str__(self) -> str:
        """
        >>> print(NacState((('X','','O'), ('','X',''), ('','X','O'))))
        A  X.O
        B  .X.
        C  .XO
        """
        return '\n'.join([chr(ord('A') + i) + '  ' + ''.join(e or '.' for e in row) for i, row in enumerate(self.board)])

@dataclass
class Nac(Game[NacState, NacAction]):
    size: int = 3
    markers: Tuple[Marker, Marker] = field(default_factory=lambda: (x_marker, o_marker))
    use_symmetry: bool = False

    def get_init_state(self) -> NacState:
        """
        >>> game = Nac()
        >>> game.get_init_state()
        NacState(board=(('', '', ''), ('', '', ''), ('', '', '')), next_player_index=0)
        """
        board = tuple(tuple(empty_square for _ in range(self.size)) for _ in range(self.size))
        return NacState(board=board)

    def get_actions(self, state: NacState) -> Generator[NacAction, None, None]:
        """
        >>> game = Nac()
        >>> state = game.get_init_state()
        >>> len(list(game.get_actions(state)))
        9
        >>> state = NacState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> list(game.get_actions(state))
        [(0, 1), (2, 0), (2, 1)]
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> list(game.get_actions(state))
        [(2, 0), (2, 1)]
        """
        if self.use_symmetry and all(s == empty_square for row in state.board for s in row):
            if self.size != 3:
                raise NotImplementedError('Use symmetry only works for size 3 currently.')
            for r, c in [(0, 0), (1, 0), (1, 1)]:
                yield NacAction(r, c)

        for r in range(self.size):
            for c in range(self.size):
                if state.board[r][c] == empty_square:
                    yield NacAction(r, c)

    def updated(self, state: NacState, action: NacAction) -> NacState:
        """
        >>> game = Nac(markers=(o_marker, x_marker))
        >>> state = game.get_init_state()
        >>> game.updated(state, NacAction(2, 1))
        NacState(board=(('', '', ''), ('', '', ''), ('', 'O', '')), next_player_index=1)
        >>> state = NacState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.updated(state, NacAction(0, 1))
        NacState(board=(('X', 'O', 'O'), ('X', 'O', 'O'), ('', '', 'X')), next_player_index=1)
        """
        # pylint: disable=no-self-use
        new_board = [list(row) for row in state.board]
        new_board[action.row][action.col] = self.markers[state.next_player_index]
        return NacState(
            tuple(tuple(row) for row in new_board),
            next_player_index = 1 - state.next_player_index,
        )

    def _get_winner(self, state: NacState) -> Optional[PlayerIndex]:
        """
        >>> game = Nac()
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game._get_winner(state)
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game._get_winner(state)
        1
        """
        b, size = state.board, self.size
        if size > 3:
            raise NotImplementedError('Only max size 3 boards implemented.')
        for m in (x_marker, o_marker):
            if any(all(b[r][c] == m for c in range(size)) for r in range(size)) or \
            any(all(b[r][c] == m for r in range(size)) for c in range(size)) or \
            all(b[d][d] == m for d in range(size)) or \
            all(b[d][size - 1 - d] == m for d in range(size)):
                return 0 if m == self.markers[0] else 1
        return None

    def get_score_and_game_over(self, state: NacState) -> Tuple[PlayerIndex, bool]:
        """
        Because this is a two player game, the last player to take a turn was "not" state.next_player_index.
        >>> game = Nac()
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.get_score_and_game_over(state)
        (0, False)
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game.get_score_and_game_over(state)  # O is the winner; X is the next player
        (1, True)
        >>> state = NacState(state.board, next_player_index=1)
        >>> game.get_score_and_game_over(state)
        (-1, True)
        """
        winner = self._get_winner(state)
        if winner is not None:
            if winner == state.next_player_index:
                return -1, True
            return 1, True
        return 0, all(s for row in state.board for s in row)
