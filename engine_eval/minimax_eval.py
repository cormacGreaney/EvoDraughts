"""
Head-to-head evaluation: evolved 8x8 strategies vs the minimax engine.

Strategies are executed with the same position-evaluation move selection as
``evolve_draughts_8x8`` (and ``reevaluate_strategies_match_evolution``), not the
legacy direct move-index interpretation in ``play_against_evolved``.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

# Default batch for the report: fixed set of 8x8 strategies (under results/).
REPORT_MINIMAX_STRATEGIES: Tuple[str, ...] = (
    "evolution_8x8_20260316_010459.txt",
    "evolution_8x8_20260219_203646.txt",
    "evolution_8x8_20251218_002100.txt",
    "evolution_8x8_20260312_214315.txt",
    "evolution_8x8_20260227_011809.txt",
)

# Project root must be on path before engine_eval / draughts imports when run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from engine_eval.minimax_engine.board import Board
from engine_eval.minimax_engine.constants import RED, WHITE
from engine_eval.minimax_engine.piece import Piece

from draughts_game import DraughtsBoard
from play_against_evolved import load_strategy_from_file

import evolve_draughts_8x8


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

    strategy = evolve_draughts_8x8.strategy_from_phenotype(phenotype)

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


def _write_minimax_report_files(
    output_dir: Path,
    stamp: str,
    rows: Sequence[Tuple[str, int, int, int]],
    games: int,
    depth: int,
) -> Tuple[Path, Path]:
    """
    Write human-readable summary and CSV. rows: (filename, wins, draws, losses).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"minimax_eval_{stamp}.txt"
    csv_path = output_dir / f"minimax_eval_{stamp}.csv"

    lines: List[str] = [
        "EvoDraughts — Minimax head-to-head evaluation",
        "=" * 60,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Games per strategy: {games}",
        f"Minimax search depth: {depth}",
        "Strategy execution: evolve_draughts_8x8 (position evaluation per legal move)",
        "",
        "Note on outcomes:",
        "  Play is deterministic: same opening and fixed-depth minimax each time.",
        "  Alternating colours splits games into two identical halves (as White vs as Red);",
        "  each half repeats the same trajectory, so aggregate W/D/L counts take at most",
        "  two outcome types. Running many games per strategy is for verification and",
        "  reporting, not independent random samples.",
        "",
        "Results",
        "-" * 60,
        f"{'Strategy file':<48} {'W':>5} {'D':>5} {'L':>5}  Win rate",
        "-" * 60,
    ]

    csv_rows: List[dict] = []
    for name, w, d, l in rows:
        total = w + d + l
        rate = (w / total) if total else 0.0
        lines.append(f"{name:<48} {w:5d} {d:5d} {l:5d}  {rate:.1%}")
        csv_rows.append(
            {
                "strategy_file": name,
                "wins": w,
                "draws": d,
                "losses": l,
                "games": total,
                "win_rate": round(rate, 6),
            }
        )

    lines.append("")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=["strategy_file", "wins", "draws", "losses", "games", "win_rate"],
        )
        wcsv.writeheader()
        wcsv.writerows(csv_rows)

    return txt_path, csv_path


def main() -> None:
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 8x8 evolved strategies against the minimax engine, using the same "
            "position-evaluation strategy semantics as evolve_draughts_8x8."
        )
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
    parser.add_argument(
        "--all-8x8",
        action="store_true",
        help=(
            "Evaluate every results/evolution_8x8_*.txt instead of the default "
            "report shortlist."
        ),
    )
    parser.add_argument(
        "--include-minimax-trained",
        action="store_true",
        help=(
            "With --all-8x8: also include engine_eval/minimax_results/evolution_minimax_8x8_*.txt."
        ),
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Evaluate a single result file path instead of the default batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for minimax_eval_<timestamp>.txt and .csv (default: project results/).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout only; do not write report files.",
    )

    args = parser.parse_args()

    results_dir = project_root / "results"
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = results_dir

    if args.results_file:
        files = [Path(args.results_file)]
        for p in files:
            if not p.is_file():
                print(f"File not found: {p}")
                return
    else:
        if not results_dir.exists():
            print(f"No results directory found at {results_dir}")
            return

        if args.all_8x8:
            files = sorted(results_dir.glob("evolution_8x8_*.txt"))
            if args.include_minimax_trained:
                mm_dir = project_root / "engine_eval" / "minimax_results"
                if mm_dir.is_dir():
                    files.extend(sorted(mm_dir.glob("evolution_minimax_8x8_*.txt")))
                files = sorted(set(files), key=lambda p: p.name)
        else:
            files = []
            missing: List[str] = []
            for fname in REPORT_MINIMAX_STRATEGIES:
                p = results_dir / fname
                if p.is_file():
                    files.append(p)
                else:
                    missing.append(fname)
            if missing:
                print("Missing curated strategy file(s) under results/:")
                for m in missing:
                    print(f"  {m}")
                return

        if not files:
            print("No matching 8x8 evolution result files found.")
            return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Evaluating {len(files)} strategies against minimax-based engine (depth={args.depth}).")
    print()

    rows: List[Tuple[str, int, int, int]] = []
    for path in files:
        wins, draws, losses = evaluate_strategy_against_minimax(
            str(path),
            games=args.games,
            depth=args.depth,
        )
        name = path.name
        rows.append((name, wins, draws, losses))
        print(f"{name}: W={wins}, D={draws}, L={losses}")

    if not args.no_save:
        txt_path, csv_path = _write_minimax_report_files(
            out_dir,
            stamp,
            rows,
            games=args.games,
            depth=args.depth,
        )
        print()
        print(f"Wrote {txt_path}")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

