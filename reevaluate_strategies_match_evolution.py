"""
Re-evaluate evolved strategies using the SAME move-selection logic as the evolution scripts.

This mirrors reevaluate_strategies.py (same seeds, games/seed, CSV + summary layout) but:
  - 8x8: uses evolve_draughts_8x8.strategy_from_phenotype (position evaluation per legal move)
  - 6x6 improved runs: evolve_draughts2.strategy_from_phenotype  
  - 6x6 non-improved runs: evolve_draughts.strategy_from_phenotype (direct eval -> move index)

Output files are named standardized_test_results_evolution_match_<timestamp>.csv and
standardized_test_summary_evolution_match_<timestamp>.txt so they do not overwrite the
original re-evaluation outputs.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import datetime
import csv

import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
grape_path = os.path.join(PROJECT_ROOT, "grape")
if grape_path not in sys.path:
    sys.path.insert(0, grape_path)

from draughts_game import play_game

# Same protocol as evolution scripts / reevaluate_strategies.py
try:
    from config import TEST_SEEDS_8x8, TEST_SEEDS_6x6, TEST_GAMES_PER_SEED
except ImportError:
    TEST_SEEDS_8x8 = [888, 889, 890, 891, 892]
    TEST_SEEDS_6x6 = [999, 1000, 1001, 1002, 1003]
    TEST_GAMES_PER_SEED = 30

# Re-use result file parsing (no edits to that file)
from reevaluate_strategies import parse_result_file

# Evolution-time strategy builders (must match training / in-run post-test)
import evolve_draughts
import evolve_draughts2
import evolve_draughts_8x8


def random_strategy(features, valid_moves):
    """Random baseline — same as evolution scripts."""
    import random

    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def _is_improved_6x6_result_file(filepath: str) -> bool:
    """6x6 runs from evolve_draughts2.py write 'Improved Version' in the header."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            head = f.read(800)
        return "Improved Version" in head
    except OSError:
        return False


def strategy_from_phenotype_like_evolution(phenotype: str, board_size: int, filepath: str):
    """
    Return a strategy callable (features, valid_moves) -> move_index,
    using the same implementation as the evolution run that produced that result file.
    """
    if board_size == 8:
        return evolve_draughts_8x8.strategy_from_phenotype(phenotype)
    if board_size == 6:
        if _is_improved_6x6_result_file(filepath):
            return evolve_draughts2.strategy_from_phenotype(phenotype)
        return evolve_draughts.strategy_from_phenotype(phenotype)
    raise ValueError(f"Unsupported board_size: {board_size}")


def evaluate_strategy_multi_seed(
    phenotype: str,
    board_size: int,
    filepath: str,
    test_seeds: list,
    games_per_seed: int,
):
    """
    Same loop as evolve_draughts_8x8 end-of-run test and reevaluate_strategies.py.
    """
    strategy = strategy_from_phenotype_like_evolution(phenotype, board_size, filepath)
    max_moves = 200 if board_size == 8 else 100
    wins_per_seed = []
    total_wins = 0
    total_games = 0

    for test_seed in test_seeds:
        import random

        random.seed(test_seed)
        np.random.seed(test_seed)
        seed_wins = 0
        for _ in range(games_per_seed):
            result = play_game(strategy, random_strategy, max_moves=max_moves, board_size=board_size)
            if result == 1:
                seed_wins += 1
                total_wins += 1
            total_games += 1
        wins_per_seed.append(seed_wins / games_per_seed)

    mean_wr = total_wins / total_games if total_games > 0 else 0.0
    std_seeds = float(np.std(wins_per_seed)) if len(wins_per_seed) > 1 else 0.0
    n = total_games
    z = 1.96
    margin = z * np.sqrt(mean_wr * (1 - mean_wr) / n) if n > 0 else 0
    ci_lo = max(0, mean_wr - margin)
    ci_hi = min(1, mean_wr + margin)

    return {
        "total_wins": total_wins,
        "total_games": total_games,
        "mean_wr": mean_wr,
        "std_seeds": std_seeds,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wins_per_seed": wins_per_seed,
    }


def main() -> None:
    print("=" * 80)
    print("RE-EVALUATING STRATEGIES (EVOLUTION-MATCHING MOVE SELECTION)")
    print("=" * 80)
    print()
    print(
        "8x8 uses evolve_draughts_8x8.strategy_from_phenotype; "
        "6x6 improved uses evolve_draughts2; "
        "6x6 classic uses evolve_draughts."
    )
    print()

    results_dir = Path(PROJECT_ROOT) / "results"
    if not results_dir.is_dir():
        print(f"No results directory: {results_dir}")
        return

    result_files = list(results_dir.glob("evolution_*.txt"))
    if not result_files:
        print("No evolution_*.txt result files found in results/")
        return

    print(f"Found {len(result_files)} evolution result files")
    print()

    parsed_results = []
    for filepath in result_files:
        try:
            result = parse_result_file(str(filepath))
            if result["strategy"] and result["board_size"]:
                result["filepath"] = str(filepath)
                parsed_results.append(result)
                print(f"Parsed: {result['file']}")
            else:
                print(f"Skipped {result['file']}: missing strategy or board size")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

    if not parsed_results:
        print("No valid results to re-evaluate")
        return

    print()
    print("=" * 80)
    print("RE-EVALUATING STRATEGIES")
    print("=" * 80)
    print()

    test_seeds_6x6 = TEST_SEEDS_6x6
    test_seeds_8x8 = TEST_SEEDS_8x8
    games_per_seed = TEST_GAMES_PER_SEED

    re_evaluated_main: list = []
    for i, result in enumerate(parsed_results, 1):
        print(f"[{i}/{len(parsed_results)}] Re-evaluating {result['file']}...", end=" ", flush=True)
        board_size = result["board_size"]
        test_seeds = test_seeds_6x6 if board_size == 6 else test_seeds_8x8
        fp = result.get("filepath") or str(results_dir / result["file"])

        try:
            out = evaluate_strategy_multi_seed(
                result["strategy"],
                board_size,
                fp,
                test_seeds,
                games_per_seed,
            )
            result["new_test_wins"] = out["total_wins"]
            result["new_test_total_games"] = out["total_games"]
            result["new_test_win_rate"] = out["mean_wr"]
            result["new_test_ci_lo"] = out["ci_lo"]
            result["new_test_ci_hi"] = out["ci_hi"]
            result["new_test_std_seeds"] = out["std_seeds"]
            result["new_test_wins_per_seed"] = out["wins_per_seed"]
            result["test_seeds"] = test_seeds
            result["test_games_per_seed"] = games_per_seed
            if result.get("fitness") is not None:
                result["train_win_rate"] = 1.0 - result["fitness"]
                result["train_test_gap"] = result["train_win_rate"] - out["mean_wr"]
            else:
                result["train_win_rate"] = None
                result["train_test_gap"] = None
            result["source"] = "main"
            re_evaluated_main.append(result)
            print(
                f"OK {out['total_wins']}/{out['total_games']} -> "
                f"{out['mean_wr']:.1%} [CI {out['ci_lo']:.1%}-{out['ci_hi']:.1%}]"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    re_evaluated_minimax: list = []
    minimax_dir = Path(PROJECT_ROOT) / "engine_eval" / "minimax_results"
    if minimax_dir.exists():
        minimax_files = list(minimax_dir.glob("evolution_minimax_8x8_*.txt"))
        if minimax_files:
            print()
            print("=" * 80)
            print("RE-EVALUATING MINIMAX-TARGETED 8x8 STRATEGIES")
            print("=" * 80)
            print()
            for i, filepath in enumerate(minimax_files, 1):
                print(
                    f"[minimax {i}/{len(minimax_files)}] Re-evaluating {filepath.name}...",
                    end=" ",
                    flush=True,
                )
                try:
                    result = parse_result_file(str(filepath))
                    if not result["strategy"] or result["board_size"] != 8:
                        print("Skipped (missing strategy or not 8x8)")
                        continue
                    fp = str(filepath)
                    out = evaluate_strategy_multi_seed(
                        result["strategy"],
                        8,
                        fp,
                        test_seeds_8x8,
                        games_per_seed,
                    )
                    result["filepath"] = fp
                    result["new_test_wins"] = out["total_wins"]
                    result["new_test_total_games"] = out["total_games"]
                    result["new_test_win_rate"] = out["mean_wr"]
                    result["new_test_ci_lo"] = out["ci_lo"]
                    result["new_test_ci_hi"] = out["ci_hi"]
                    result["new_test_std_seeds"] = out["std_seeds"]
                    result["new_test_wins_per_seed"] = out["wins_per_seed"]
                    result["test_seeds"] = test_seeds_8x8
                    result["test_games_per_seed"] = games_per_seed
                    if result.get("fitness") is not None:
                        result["train_win_rate"] = 1.0 - result["fitness"]
                        result["train_test_gap"] = result["train_win_rate"] - out["mean_wr"]
                    else:
                        result["train_win_rate"] = None
                        result["train_test_gap"] = None
                    result["source"] = "minimax"
                    re_evaluated_minimax.append(result)
                    print(
                        f"OK {out['total_wins']}/{out['total_games']} -> "
                        f"{out['mean_wr']:.1%} [CI {out['ci_lo']:.1%}-{out['ci_hi']:.1%}]"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")

    re_evaluated = re_evaluated_main + re_evaluated_minimax
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = results_dir / f"standardized_test_results_evolution_match_{timestamp}.csv"
    report_filename = results_dir / f"standardized_test_summary_evolution_match_{timestamp}.txt"

    with open(csv_filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Result File",
                "Date",
                "Board Size",
                "Population",
                "Generations",
                "Games per Eval",
                "Co-evolution",
                "Strategy Phenotype",
                "Training Fitness",
                "Train Win Rate",
                "New Test Wins/Total",
                "Test Win Rate",
                "95% CI Lo",
                "95% CI Hi",
                "Variance Across Seeds (std)",
                "Train-Test Gap",
            ]
        )
        for result in re_evaluated:
            writer.writerow(
                [
                    result["file"],
                    result["date"],
                    f"{result['board_size']}x{result['board_size']}",
                    result["population"],
                    result["generations"],
                    result["games_per_eval"],
                    result["coevolution"],
                    (result["strategy"] or "")[:80],
                    result["fitness"] if result.get("fitness") is not None else "",
                    f"{result['train_win_rate']:.2f}" if result.get("train_win_rate") is not None else "",
                    f"{result['new_test_wins']}/{result['new_test_total_games']}",
                    f"{result['new_test_win_rate']:.4f}",
                    f"{result['new_test_ci_lo']:.4f}",
                    f"{result['new_test_ci_hi']:.4f}",
                    f"{result['new_test_std_seeds']:.4f}",
                    f"{result['train_test_gap']:.4f}" if result.get("train_test_gap") is not None else "",
                ]
            )

    print(f"Saved standardized test results to: {csv_filename}")

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STANDARDIZED TEST EVALUATION SUMMARY (EVOLUTION-MATCHING STRATEGY)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Strategy interpretation: same as evolve_draughts_8x8 / evolve_draughts2 / evolve_draughts\n")
        f.write(f"Test Protocol:\n")
        f.write(f"  - 6x6: seeds={test_seeds_6x6}, {games_per_seed} games/seed\n")
        f.write(f"  - 8x8: seeds={test_seeds_8x8}, {games_per_seed} games/seed\n")
        f.write(f"  - Opponent: random baseline (fixed set per seed)\n\n")

        results_6x6 = [r for r in re_evaluated if r["board_size"] == 6]
        results_8x8_main = [r for r in re_evaluated if r["board_size"] == 8 and r.get("source") == "main"]
        results_8x8_minimax = [r for r in re_evaluated if r["board_size"] == 8 and r.get("source") == "minimax"]

        if results_6x6:
            f.write("=" * 80 + "\n")
            f.write("6x6 STRATEGIES\n")
            f.write("=" * 80 + "\n\n")
            for result in sorted(results_6x6, key=lambda x: x["new_test_win_rate"], reverse=True):
                f.write(f"File: {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                f.write(f"  Training Fitness: {result['fitness']:.4f}\n")
                f.write(
                    f"  Test: {result['new_test_wins']}/{result['new_test_total_games']} -> "
                    f"{result['new_test_win_rate']:.1%} [95% CI: {result['new_test_ci_lo']:.1%}-{result['new_test_ci_hi']:.1%}]\n"
                )
                f.write(f"  Variance across seeds (std): {result['new_test_std_seeds']:.3f}\n")
                if result.get("train_test_gap") is not None:
                    f.write(f"  Train-test gap: {result['train_test_gap']:.1%}\n")
                f.write("\n")
            win_rates = [r["new_test_win_rate"] for r in results_6x6]
            f.write(f"6x6 Statistics:\n")
            f.write(f"  Mean Win Rate: {np.mean(win_rates):.1%}\n")
            f.write(f"  Std Dev: {np.std(win_rates):.1%}\n")
            f.write(f"  Min: {np.min(win_rates):.1%}, Max: {np.max(win_rates):.1%}\n")
            f.write(f"  Number of runs: {len(results_6x6)}\n\n")

        if results_8x8_main:
            f.write("=" * 80 + "\n")
            f.write("8x8 STRATEGIES (MAIN RUNS)\n")
            f.write("=" * 80 + "\n\n")
            for result in sorted(results_8x8_main, key=lambda x: x["new_test_win_rate"], reverse=True):
                f.write(f"File: {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                f.write(f"  Training Fitness: {result['fitness']:.4f}\n")
                f.write(
                    f"  Test: {result['new_test_wins']}/{result['new_test_total_games']} -> "
                    f"{result['new_test_win_rate']:.1%} [95% CI: {result['new_test_ci_lo']:.1%}-{result['new_test_ci_hi']:.1%}]\n"
                )
                f.write(f"  Variance across seeds (std): {result['new_test_std_seeds']:.3f}\n")
                if result.get("train_test_gap") is not None:
                    f.write(f"  Train-test gap: {result['train_test_gap']:.1%}\n")
                f.write("\n")
            win_rates = [r["new_test_win_rate"] for r in results_8x8_main]
            f.write(f"8x8 Statistics (Main Runs):\n")
            f.write(f"  Mean Win Rate: {np.mean(win_rates):.1%}\n")
            f.write(f"  Std Dev: {np.std(win_rates):.1%}\n")
            f.write(f"  Min: {np.min(win_rates):.1%}, Max: {np.max(win_rates):.1%}\n")
            f.write(f"  Number of runs: {len(results_8x8_main)}\n\n")

        if results_8x8_minimax:
            f.write("=" * 80 + "\n")
            f.write("8x8 STRATEGIES (MINIMAX-TARGETED RUNS)\n")
            f.write("=" * 80 + "\n\n")
            for result in sorted(results_8x8_minimax, key=lambda x: x["new_test_win_rate"], reverse=True):
                f.write(f"File: {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                f.write(f"  Training Fitness: {result['fitness']:.4f}\n")
                f.write(
                    f"  Test: {result['new_test_wins']}/{result['new_test_total_games']} -> "
                    f"{result['new_test_win_rate']:.1%} [95% CI: {result['new_test_ci_lo']:.1%}-{result['new_test_ci_hi']:.1%}]\n"
                )
                f.write(f"  Variance across seeds (std): {result['new_test_std_seeds']:.3f}\n")
                if result.get("train_test_gap") is not None:
                    f.write(f"  Train-test gap: {result['train_test_gap']:.1%}\n")
                f.write("\n")
            win_rates_mm = [r["new_test_win_rate"] for r in results_8x8_minimax]
            f.write(f"8x8 Statistics (Minimax-Targeted):\n")
            f.write(f"  Mean Win Rate: {np.mean(win_rates_mm):.1%}\n")
            f.write(f"  Std Dev: {np.std(win_rates_mm):.1%}\n")
            f.write(f"  Min: {np.min(win_rates_mm):.1%}, Max: {np.max(win_rates_mm):.1%}\n")
            f.write(f"  Number of runs: {len(results_8x8_minimax)}\n\n")

    print(f"Saved summary report to: {report_filename}")
    print()
    print("=" * 80)
    print("RE-EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Re-evaluated {len(re_evaluated)} strategies")
    print("Results saved to:")
    print(f"  - {csv_filename}")
    print(f"  - {report_filename}")


if __name__ == "__main__":
    main()
