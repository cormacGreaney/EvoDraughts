from .constants import RED, WHITE


class Piece:
    def __init__(self, row: int, col: int, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False

    def make_king(self):
        self.king = True

    def move(self, row: int, col: int):
        self.row = row
        self.col = col

    def __repr__(self):
        return "W" if self.color == WHITE else "R"

