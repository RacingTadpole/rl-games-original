# pylint: disable=unsubscriptable-object

# Noughts and crosses

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, Sequence, cast, Generic, TypeVar
from copy import deepcopy
from .game import Game

Marker = Literal['X', 'O']
Square = Literal['X', 'O', '']

x_marker, o_marker, empty_square = cast(Marker, 'X'), cast(Marker, 'O'), cast(Square, '')

NacAction = Tuple[int, int]


@dataclass()
class NacState:
    board: Tuple[Tuple[Square, ...], ...] = ()
    next_go: Marker = x_marker


@dataclass()
class Nac(Game[NacState, NacAction]):
    size: int = 3
    use_symmetry: bool = False

    def get_actions(self, state):
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
            if state.board.size != 3:
                raise NotImplementedError('Use symmetry only works for size 3 currently.')
            for r, c in [(0, 0), (1, 0), (1, 1)]:
                yield (r, c)

        for r in range(self.size):
            for c in range(self.size):
                if state.board[r][c] == empty_square:
                    yield (r, c)

    def get_init_state(self, next_go=x_marker) -> NacState:
        """
        >>> game = Nac()
        >>> game.get_init_state()
        NacState(board=(('', '', ''), ('', '', ''), ('', '', '')), next_go='X')
        """
        return NacState(board=((empty_square,) * self.size,) * self.size, next_go=next_go)

    def updated(self, state: NacState, action: NacAction) -> NacState:
        """
        >>> game = Nac()
        >>> state = game.get_init_state(next_go=o_marker)
        >>> game.updated(state, (2, 1))
        NacState(board=(('', '', ''), ('', '', ''), ('', 'O', '')), next_go='X')
        >>> state = NacState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.updated(state, (0, 1))
        NacState(board=(('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')), next_go='O')
        """
        new_board = [list(row) for row in state.board]
        new_board[action[0]][action[1]] = state.next_go
        return NacState(
            tuple(tuple(row) for row in new_board),
            next_go='X' if state.next_go == 'O' else 'O',
        )

    def _get_winner(self, state: NacState) -> Optional[Marker]:
        """
        >>> game = Nac()
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game._get_winner(state)
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game._get_winner(state)
        'O'
        """
        b, size = state.board, self.size
        for m in (x_marker, o_marker):
            if any(all(b[r][c] == m for c in range(size)) for r in range(size)) or \
            any(all(b[r][c] == m for r in range(size)) for c in range(size)) or \
            all(b[d][d] == m for d in range(size)) or \
            all(b[d][size - 1 - d] == m for d in range(size)):
                return m
        return None

    def get_score_and_game_over(self, state: NacState) -> Tuple[int, bool]:
        """
        >>> game = Nac()
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X')))
        >>> game.get_score_and_game_over(state)
        (0, False)
        >>> state = NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X')))
        >>> game.get_score_and_game_over(state)  # O is the winner; X is the next player
        (1, True)
        >>> state = NacState(state.board, next_go=o_marker)
        >>> game.get_score_and_game_over(state)
        (-1, True)
        """
        winner = self._get_winner(state)
        if winner is not None:
            if winner == state.next_go:
                return -1, True
            return 1, True
        return 0, all(s for row in state.board for s in row)