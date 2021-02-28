# pylint: disable=unsubscriptable-object

# Noughts and crosses

import random
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Iterator, Dict, List, Sequence, cast, Generic, TypeVar
from copy import deepcopy
from .game import GameState

Marker = Literal['X', 'O']
Square = Literal['X', 'O', '']

x_marker, o_marker, empty_square = cast(Marker, 'X'), cast(Marker, 'O'), cast(Square, '')

NacAction = Tuple[int, int]


@dataclass()
class NacState:
    """
    >>> NacState()
    NacState(board=(('', '', ''), ('', '', ''), ('', '', '')), size=3, next_go='X')
    >>> NacState(size=2, next_go=o_marker)
    NacState(board=(('', ''), ('', '')), size=2, next_go='O')
    >>> NacState((('X', ''), ('X', 'O')))
    NacState(board=(('X', ''), ('X', 'O')), size=2, next_go='O')
    """
    board: Tuple[Tuple[Square, ...], ...] = ()
    size: int = 3
    # The post-init function ensures the type is Marker.
    next_go: Marker = None  # type: ignore

    def __post_init__(self):
        """
        The post-init provides useful defaults (ie. a size 3 empty board with X going first),
        if any of the inputs are not specified.
        The size is always overridden with the length of the board.
        """
        if len(self.board) == 0:
            self.board = ((empty_square,) * self.size,) * self.size
            if self.next_go is None:
                self.next_go = x_marker
            return
        self.size = len(self.board)
        if self.next_go is None:
            count = {x_marker: 0, o_marker: 0}
            for r in range(self.size):
                for c in range(self.size):
                    contents = self.board[r][c]
                    if contents != empty_square:
                        count[contents] += 1
            if count[x_marker] == count[o_marker] + 1:
                self.next_go = o_marker
            else:
                self.next_go = x_marker


@dataclass()
class Nac(GameState[NacState, NacAction]):
    use_symmetry: bool = False

    def get_actions(self):
        """
        >>> game = Nac(NacState())
        >>> len(list(game.get_actions()))
        9
        >>> game = Nac(NacState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X'))))
        >>> list(game.get_actions())
        [(0, 1), (2, 0), (2, 1)]
        >>> game = Nac(NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X'))))
        >>> list(game.get_actions())
        [(2, 0), (2, 1)]
        """
        if self.use_symmetry and all(s == empty_square for row in self.state.board for s in row):
            if self.state.board.size != 3:
                raise NotImplementedError('Use symmetry only works for size 3 currently.')
            for r, c in [(0, 0), (1, 0), (1, 1)]:
                yield (r, c)

        for r in range(self.state.size):
            for c in range(self.state.size):
                if self.state.board[r][c] == empty_square:
                    yield (r, c)

    def update(self, action: NacAction):
        """
        >>> game = Nac(NacState(next_go=o_marker))
        >>> game.update((2, 1))
        >>> game.state.board
        (('', '', ''), ('', '', ''), ('', 'O', ''))
        >>> game = Nac(NacState((('X', '', 'O'), ('X', 'O', 'O'), ('', '', 'X'))))
        >>> game.update((0, 1))
        >>> game.state.board
        (('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X'))
        """
        new_board = [list(row) for row in self.state.board]
        new_board[action[0]][action[1]] = self.state.next_go
        self.state = NacState(tuple(tuple(row) for row in new_board))

    def _get_winner(self) -> Optional[Marker]:
        """
        >>> game = Nac(NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('', '', 'X'))))
        >>> game._get_winner()
        >>> game = Nac(NacState((('X', 'X', 'O'), ('X', 'O', 'O'), ('O', '', 'X'))))
        >>> game._get_winner()
        'O'
        """
        b, size = self.state.board, self.state.size
        for m in (x_marker, o_marker):
            if any(all(b[r][c] == m for c in range(size)) for r in range(size)) or \
            any(all(b[r][c] == m for r in range(size)) for c in range(size)) or \
            all(b[d][d] == m for d in range(size)) or \
            all(b[d][size - 1 - d] == m for d in range(size)):
                return m
        return None

    def get_score_and_game_over(self) -> Tuple[int, bool]:
        winner = self._get_winner()
        if winner is not None:
            if winner == self.state.next_go:
                return -1, True
            return 1, True
        return 0, all(s for row in self.state.board for s in row)

