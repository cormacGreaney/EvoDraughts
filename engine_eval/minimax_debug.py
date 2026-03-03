import argparse
import os
import sys
from typing import List, Tuple

from engine_eval.minimax_engine.board import Board
from engine_eval.minimax_engine.constants import RED, WHITE, ROWS, COLS
from engine_eval.minimax_engine.piece import Piece
from engine_eval.minimax_eval import (
    _get_all_moves,
    _strategy_move,
    _minimax_choose_move,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play_against_evolved import load_strategy_from_file, strategy_from_phenotype


def _print_board(board: Board) -> None:
    symbols = {("W", False): "w", ("W", True): "W", ("R", False): "r", ("R", True): "R"}
    for row in range(ROWS):
        line = []
        for col in range(COLS):
            piece = board.get_piece(row, col)
            if piece == 0:
                line.append(".")
            else:
                key = ("W" if piece.color == WHITE else "R", piece.king)
                line.append(symbols[key])
        print("".join(line))
    print()


def _side_name(color) -> str:
    return "White" if color == WHITE else "Red"


def play_debug_game(strategy, strategy_as_white: bool, depth: int = 2, max_moves: int = 200) -> None:
    board = Board()
    current_color = WHITE
    moves_played = 0

    print("Initial position (White at top, Red at bottom):")
    _print_board(board)

    while moves_played < max_moves:
        strategy_turn = (strategy_as_white and current_color == WHITE) or (
            (not strategy_as_white) and current_color == RED
        )

        side = _side_name(current_color)
        print(f"Move {moves_played + 1}: {side} to move "
              f"({'strategy' if strategy_turn else 'engine'})")

        moves = _get_all_moves(board, current_color)
        if not moves:
            print(f"No legal moves for {side}. Game over.")
            break

        if strategy_turn:
            move = _strategy_move(board, strategy, strategy_is_white=strategy_as_white)
        else:
            move = _minimax_choose_move(board, color=current_color, depth=depth)

        if move is None:
            print(f"{side} returned no move despite having legal moves. Game over.")
            break

        piece, dest, skipped = move
        r_to, c_to = dest
        print(
            f"{side} moves from ({piece.row}, {piece.col}) to ({r_to}, {c_to}); "
            f"captures {len(skipped)} piece(s)."
        )

        board.move(piece, r_to, c_to)
        if skipped:
            board.remove(skipped)

        moves_played += 1

        print("Board after move:")
        _print_board(board)

        winner = board.winner()
        if winner is not None:
            winner_side = _side_name(winner)
            print(f"Winner by piece elimination: {winner_side}")
            break

        current_color = RED if current_color == WHITE else WHITE

    if moves_played >= max_moves:
        print(f"Reached max moves ({max_moves}) without a winner.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug one or more games against the minimax-based engine with full move trace."
    )
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to an 8x8 evolution results file.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2,
        help="Number of games to play for debugging.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Minimax search depth for the engine.",
    )

    args = parser.parse_args()

    results_path = os.path.abspath(args.results_file)
    phenotype, board_size, fitness, metadata = load_strategy_from_file(results_path)
    if board_size != 8:
        raise ValueError("Expected an 8x8 strategy results file.")

    strategy = strategy_from_phenotype(phenotype, board_size)

    for i in range(args.games):
        as_white = (i % 2 == 0)
        print("=" * 60)
        print(f"Game {i + 1} (strategy as {'White' if as_white else 'Red'}):")
        print("=" * 60)
        play_debug_game(strategy, strategy_as_white=as_white, depth=args.depth)
        print()


if __name__ == "__main__":
    main()

