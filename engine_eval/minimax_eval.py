import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from engine_eval.minimax_engine.board import Board
from engine_eval.minimax_engine.constants import RED, WHITE
from engine_eval.minimax_engine.piece import Piece

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draughts_game import DraughtsBoard
from play_against_evolved import load_strategy_from_file, strategy_from_phenotype


def _build_draughts_board_from_minimax(board: Board, strategy_is_white: bool) -> DraughtsBoard:
    db = DraughtsBoard(board_size=8)
    db.board[:, :] = 0

    for row in range(8):
        for col in range(8):
            piece = board.get_piece(row, col)
            if piece == 0:
                continue
            if piece.color == WHITE:
                db.board[row, col] = 2 if piece.king else 1
            else:
                db.board[row, col] = -2 if piece.king else -1

    db.current_player = 1 if strategy_is_white else -1
    db.game_over = False
    db.winner = None
    return db


def _get_all_moves(board: Board, color) -> List[Tuple[Piece, Tuple[int, int], List[Piece]]]:
    moves: List[Tuple[Piece, Tuple[int, int], List[Piece]]] = []
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for (r, c), skipped in valid_moves.items():
            moves.append((piece, (r, c), skipped))
    return moves


def _clone_and_apply(board: Board, piece: Piece, dest: Tuple[int, int], skipped: List[Piece]) -> Board:
    new_board = board.clone()
    src_piece = new_board.get_piece(piece.row, piece.col)
    r, c = dest
    new_board.move(src_piece, r, c)
    if skipped:
        to_remove: List[Piece] = []
        for s in skipped:
            p = new_board.get_piece(s.row, s.col)
            if isinstance(p, Piece):
                to_remove.append(p)
        if to_remove:
            new_board.remove(to_remove)
    return new_board


def _evaluate_for_color(board: Board, color) -> float:
    value = board.evaluate()
    if color == WHITE:
        return value
    return -value


def _minimax_choose_move(board: Board, color, depth: int) -> Tuple[Piece, Tuple[int, int], List[Piece]] | None:
    moves = _get_all_moves(board, color)
    if not moves:
        return None

    best_score = None
    best_move = moves[0]

    for piece, dest, skipped in moves:
        child = _clone_and_apply(board, piece, dest, skipped)
        score = _minimax_score(child, color, depth - 1, maximizing=False)
        if best_score is None or score > best_score:
            best_score = score
            best_move = (piece, dest, skipped)

    return best_move


def _minimax_score(board: Board, color, depth: int, maximizing: bool) -> float:
    winner = board.winner()
    if depth == 0 or winner is not None:
        if winner is not None:
            if winner == WHITE:
                return float("inf") if color == WHITE else float("-inf")
            if winner == RED:
                return float("inf") if color == RED else float("-inf")
        return _evaluate_for_color(board, color)

    if maximizing:
        best = float("-inf")
        moves = _get_all_moves(board, color)
        if not moves:
            return _evaluate_for_color(board, color)
        for piece, dest, skipped in moves:
            child = _clone_and_apply(board, piece, dest, skipped)
            score = _minimax_score(child, color, depth - 1, maximizing=False)
            best = max(best, score)
        return best
    else:
        opp_color = RED if color == WHITE else WHITE
        best = float("inf")
        moves = _get_all_moves(board, opp_color)
        if not moves:
            return _evaluate_for_color(board, color)
        for piece, dest, skipped in moves:
            child = _clone_and_apply(board, piece, dest, skipped)
            score = _minimax_score(child, color, depth - 1, maximizing=True)
            best = min(best, score)
        return best


def _strategy_move(board: Board, strategy, strategy_is_white: bool):
    color = WHITE if strategy_is_white else RED
    moves = _get_all_moves(board, color)
    if not moves:
        return None

    db = _build_draughts_board_from_minimax(board, strategy_is_white=strategy_is_white)
    features = db.get_board_features()

    valid_moves_for_strategy = []
    for piece, dest, skipped in moves:
        r_from, c_from = piece.row, piece.col
        r_to, c_to = dest
        valid_moves_for_strategy.append(((r_from, c_from), (r_to, c_to)))

    idx = strategy(features, valid_moves_for_strategy)
    try:
        idx_int = int(idx)
    except Exception:
        idx_int = 0

    if idx_int < 0 or idx_int >= len(moves):
        idx_int = max(0, min(idx_int, len(moves) - 1))

    return moves[idx_int]


def _play_single_game(strategy, strategy_as_white: bool, depth: int = 2, max_moves: int = 200) -> int:
    board = Board()
    moves_played = 0
    current_color = WHITE

    while moves_played < max_moves:
        strategy_turn = (strategy_as_white and current_color == WHITE) or (
            (not strategy_as_white) and current_color == RED
        )

        if strategy_turn:
            move = _strategy_move(board, strategy, strategy_is_white=strategy_as_white)
        else:
            move = _minimax_choose_move(board, color=current_color, depth=depth)

        if move is None:
            if strategy_turn:
                return -1
            else:
                return 1

        piece, dest, skipped = move
        r, c = dest
        board.move(piece, r, c)
        if skipped:
            board.remove(skipped)

        moves_played += 1

        winner = board.winner()
        if winner is not None:
            if strategy_as_white and winner == WHITE:
                return 1
            if (not strategy_as_white) and winner == RED:
                return 1
            return -1

        current_color = RED if current_color == WHITE else WHITE

    return 0


def evaluate_strategy_against_minimax(results_file: str, games: int, depth: int = 2) -> Tuple[int, int, int]:
    phenotype, board_size, fitness, metadata = load_strategy_from_file(results_file)
    if board_size != 8:
        raise ValueError("Expected an 8x8 strategy results file.")

    strategy = strategy_from_phenotype(phenotype, board_size)

    wins = 0
    draws = 0
    losses = 0

    for i in range(games):
        as_white = (i % 2 == 0)
        result = _play_single_game(strategy, strategy_as_white=as_white, depth=depth)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

    return wins, draws, losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all 8x8 evolved strategies against a minimax-based engine."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games per strategy.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Minimax search depth for the engine.",
    )

    args = parser.parse_args()

    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = project_root / "results"

    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    files = sorted(results_dir.glob("evolution_8x8_*.txt"))
    if not files:
        print("No 8x8 evolution result files found.")
        return

    print(f"Evaluating {len(files)} strategies against minimax-based engine.")
    print()

    for path in files:
        wins, draws, losses = evaluate_strategy_against_minimax(
            str(path),
            games=args.games,
            depth=args.depth,
        )
        name = path.name
        print(f"{name}: W={wins}, D={draws}, L={losses}")


if __name__ == "__main__":
    main()

