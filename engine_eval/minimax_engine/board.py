from copy import deepcopy
from typing import Dict, List, Tuple

from .constants import BLACK, RED, ROWS, COLS, WHITE
from .piece import Piece


class Board:
    def __init__(self):
        self.board: List[List[Piece | int]] = []
        self.red_left = 12
        self.white_left = 12
        self.red_kings = 0
        self.white_kings = 0
        self._create_board()

    def _create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, WHITE))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, RED))
                    else:
                        self.board[row].append(0)
                else:
                    self.board[row].append(0)

    def evaluate(self) -> float:
        return (self.white_left - self.red_left) + 0.5 * (self.white_kings - self.red_kings)

    def get_all_pieces(self, color) -> List[Piece]:
        pieces: List[Piece] = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def move(self, piece: Piece, row: int, col: int):
        self.board[piece.row][piece.col], self.board[row][col] = (
            self.board[row][col],
            self.board[piece.row][piece.col],
        )
        piece.move(row, col)

        if row == ROWS - 1 or row == 0:
            if not piece.king:
                piece.make_king()
                if piece.color == WHITE:
                    self.white_kings += 1
                else:
                    self.red_kings += 1

    def get_piece(self, row: int, col: int):
        return self.board[row][col]

    def remove(self, pieces: List[Piece]):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece.color == RED:
                self.red_left -= 1
            else:
                self.white_left -= 1

    def winner(self):
        if self.red_left <= 0:
            return WHITE
        if self.white_left <= 0:
            return RED
        return None

    def get_valid_moves(self, piece: Piece) -> Dict[Tuple[int, int], List[Piece]]:
        moves: Dict[Tuple[int, int], List[Piece]] = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        if piece.color == RED or piece.king:
            moves.update(self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right))
        if piece.color == WHITE or piece.king:
            moves.update(self._traverse_left(row + 1, min(row + 3, ROWS), 1, piece.color, left))
            moves.update(self._traverse_right(row + 1, min(row + 3, ROWS), 1, piece.color, right))

        return moves

    def _traverse_left(
        self,
        start: int,
        stop: int,
        step: int,
        color,
        left: int,
        skipped: List[Piece] | None = None,
    ) -> Dict[Tuple[int, int], List[Piece]]:
        if skipped is None:
            skipped = []

        moves: Dict[Tuple[int, int], List[Piece]] = {}
        last: List[Piece] = []

        for r in range(start, stop, step):
            if left < 0:
                break

            current = self.board[r][left]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last

                if last:
                    if step == -1:
                        row = max(r - 3, 0)
                    else:
                        row = min(r + 3, ROWS)
                    moves.update(self._traverse_left(r + step, row, step, color, left - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, left + 1, skipped=last))
                    break
                else:
                    break
            elif current.color == color:
                break
            else:
                last = [current]

            left -= 1

        return moves

    def _traverse_right(
        self,
        start: int,
        stop: int,
        step: int,
        color,
        right: int,
        skipped: List[Piece] | None = None,
    ) -> Dict[Tuple[int, int], List[Piece]]:
        if skipped is None:
            skipped = []

        moves: Dict[Tuple[int, int], List[Piece]] = {}
        last: List[Piece] = []

        for r in range(start, stop, step):
            if right >= COLS:
                break

            current = self.board[r][right]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last

                if last:
                    if step == -1:
                        row = max(r - 3, 0)
                    else:
                        row = min(r + 3, ROWS)
                    moves.update(self._traverse_left(r + step, row, step, color, right - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, right + 1, skipped=last))
                    break
                else:
                    break
            elif current.color == color:
                break
            else:
                last = [current]

            right += 1

        return moves

    def clone(self) -> "Board":
        return deepcopy(self)

