"""
Re-evaluate all existing strategies with standardized test protocol.
This ensures fair comparison by testing all strategies against the same random opponent sequence.
"""

import sys
import os
import re
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import csv

# Add grape directory to path
grape_path = os.path.join(os.path.dirname(__file__), 'grape')
sys.path.insert(0, grape_path)

from draughts_game import DraughtsBoard, play_game
from draughts_functions import add, sub, mul, pdiv, max_, min_, greater_than_or_equal, less_than_or_equal, if_, neg, abs_

# Test protocol: multiple seeds, 30+ games per seed (from config if available)
try:
    from config import TEST_SEEDS_8x8, TEST_SEEDS_6x6, TEST_GAMES_PER_SEED
except ImportError:
    TEST_SEEDS_8x8 = [888, 889, 890, 891, 892]
    TEST_SEEDS_6x6 = [999, 1000, 1001, 1002, 1003]
    TEST_GAMES_PER_SEED = 30


def random_strategy(features, valid_moves):
    """Random baseline strategy for comparison."""
    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def strategy_from_phenotype(phenotype, board_size):
    """
    Create a strategy function from a GRAPE phenotype.
    The phenotype is evaluated to produce a move index.
    """
    def strategy(features, valid_moves):
        if not valid_moves:
            return 0
        
        try:
            # Create x array from features for easier indexing
            x = np.array(features, dtype=float)
            
            # Ensure we have enough features (pad with zeros if needed)
            expected_features = board_size * board_size + 11
            if len(x) < expected_features:
                x = np.pad(x, (0, max(0, expected_features - len(x))), 'constant')
            elif len(x) > expected_features:
                x = x[:expected_features]
            
            # Create a local namespace with features as x array
            local_vars = {
                'x': x,
                'np': np
            }
            
            # Add individual x[i] access for compatibility
            for i in range(len(x)):
                local_vars[f'x[{i}]'] = x[i]
            
            # Import functions into namespace
            local_vars.update({
                'add': add, 'sub': sub, 'mul': mul, 'pdiv': pdiv,
                'max_': max_, 'min_': min_,
                'greater_than_or_equal': greater_than_or_equal,
                'less_than_or_equal': less_than_or_equal,
                'if_': if_, 'neg': neg, 'abs_': abs_
            })
            
            # Evaluate phenotype
            result = eval(phenotype, {"__builtins__": {}}, local_vars)
            
            # Ensure result is a number and convert to move index
            try:
                move_idx = int(abs(result)) % len(valid_moves)
            except (ValueError, TypeError, OverflowError):
                move_idx = 0
            
            return move_idx
        except Exception as e:
            # If evaluation fails, return random move
            return random.randint(0, len(valid_moves) - 1) if valid_moves else 0
    
    return strategy


def parse_result_file(filepath):
    """Parse an evolution result file and extract key information."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata
    date_match = re.search(r'Date: (.+)', content)
    board_size_match = re.search(r'Board Size: (\d+)x(\d+)', content)
    pop_size_match = re.search(r'Population Size: (\d+)', content)
    gens_match = re.search(r'Generations: (\d+)', content)
    games_per_eval_match = re.search(r'Games per Evaluation: (\d+)', content)
    coevol_match = re.search(r'Co-evolution: (True|False)', content)
    
    # Extract best strategy phenotype
    strategy_match = re.search(r'Best Individual:.*?\n-+\n(.+?)\n\n', content, re.DOTALL)
    
    # Extract fitness and evolution seed
    fitness_match = re.search(r'Fitness: ([\d.]+)', content)
    evolution_seed_match = re.search(r'Evolution Seed: (\d+)', content)
    # Extract old test results if present
    old_test_match = re.search(r'Test Results.*?(\d+)/(\d+) wins', content)
    
    return {
        'file': os.path.basename(filepath),
        'filepath': filepath,
        'date': date_match.group(1) if date_match else 'Unknown',
        'board_size': int(board_size_match.group(1)) if board_size_match else None,
        'population': int(pop_size_match.group(1)) if pop_size_match else None,
        'generations': int(gens_match.group(1)) if gens_match else None,
        'games_per_eval': int(games_per_eval_match.group(1)) if games_per_eval_match else None,
        'coevolution': coevol_match.group(1) == 'True' if coevol_match else None,
        'strategy': strategy_match.group(1).strip() if strategy_match else None,
        'fitness': float(fitness_match.group(1)) if fitness_match else None,
        'evolution_seed': int(evolution_seed_match.group(1)) if evolution_seed_match else None,
        'old_test_wins': int(old_test_match.group(1)) if old_test_match else None,
        'old_test_total': int(old_test_match.group(2)) if old_test_match else None,
    }


def evaluate_strategy_multi_seed(phenotype, board_size, test_seeds, games_per_seed):
    """
    Evaluate a strategy with multiple test seeds (fixed opponent set per seed).
    Returns: dict with total_wins, total_games, mean_wr, std_seeds, ci_lo, ci_hi, wins_per_seed
    """
    strategy = strategy_from_phenotype(phenotype, board_size)
    max_moves = 200 if board_size == 8 else 100
    wins_per_seed = []
    total_wins = 0
    total_games = 0

    for test_seed in test_seeds:
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
        'total_wins': total_wins,
        'total_games': total_games,
        'mean_wr': mean_wr,
        'std_seeds': std_seeds,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'wins_per_seed': wins_per_seed,
    }


def main():
    """Main function to re-evaluate all strategies."""
    print("=" * 80)
    print("RE-EVALUATING STRATEGIES WITH STANDARDIZED TEST PROTOCOL")
    print("=" * 80)
    print()
    
    results_dir = Path('results')
    
    # Find all evolution result files
    result_files = list(results_dir.glob('evolution_*.txt'))
    
    if not result_files:
        print("No evolution result files found in results/ directory")
        return
    
    print(f"Found {len(result_files)} result files to re-evaluate")
    print()
    
    # Parse all results
    parsed_results = []
    for filepath in result_files:
        try:
            result = parse_result_file(filepath)
            if result['strategy'] and result['board_size']:
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
    
    # Re-evaluate each strategy (multi-seed protocol)
    test_seeds_6x6 = TEST_SEEDS_6x6
    test_seeds_8x8 = TEST_SEEDS_8x8
    games_per_seed = TEST_GAMES_PER_SEED
    total_games_per_strategy = len(test_seeds_8x8) * games_per_seed  # same for 6x6

    re_evaluated = []
    for i, result in enumerate(parsed_results, 1):
        print(f"[{i}/{len(parsed_results)}] Re-evaluating {result['file']}...", end=' ', flush=True)
        board_size = result['board_size']
        test_seeds = test_seeds_6x6 if board_size == 6 else test_seeds_8x8

        try:
            out = evaluate_strategy_multi_seed(
                result['strategy'], board_size, test_seeds, games_per_seed
            )
            result['new_test_wins'] = out['total_wins']
            result['new_test_total_games'] = out['total_games']
            result['new_test_win_rate'] = out['mean_wr']
            result['new_test_ci_lo'] = out['ci_lo']
            result['new_test_ci_hi'] = out['ci_hi']
            result['new_test_std_seeds'] = out['std_seeds']
            result['new_test_wins_per_seed'] = out['wins_per_seed']
            result['test_seeds'] = test_seeds
            result['test_games_per_seed'] = games_per_seed
            # Train-test gap (training win rate = 1 - fitness)
            if result.get('fitness') is not None:
                result['train_win_rate'] = 1.0 - result['fitness']
                result['train_test_gap'] = result['train_win_rate'] - out['mean_wr']
            else:
                result['train_win_rate'] = None
                result['train_test_gap'] = None
            re_evaluated.append(result)
            print(f"OK {out['total_wins']}/{out['total_games']} -> {out['mean_wr']:.1%} [CI {out['ci_lo']:.1%}-{out['ci_hi']:.1%}]")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = results_dir / f'standardized_test_results_{timestamp}.csv'
    
    with open(csv_filename, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Result File', 'Date', 'Board Size', 'Population', 'Generations',
            'Games per Eval', 'Co-evolution', 'Strategy Phenotype',
            'Training Fitness', 'Train Win Rate', 'New Test Wins/Total', 'Test Win Rate',
            '95% CI Lo', '95% CI Hi', 'Variance Across Seeds (std)', 'Train-Test Gap'
        ])
        for result in re_evaluated:
            writer.writerow([
                result['file'],
                result['date'],
                f"{result['board_size']}x{result['board_size']}",
                result['population'],
                result['generations'],
                result['games_per_eval'],
                result['coevolution'],
                (result['strategy'] or '')[:80],
                result['fitness'] if result.get('fitness') is not None else '',
                f"{result['train_win_rate']:.2f}" if result.get('train_win_rate') is not None else '',
                f"{result['new_test_wins']}/{result['new_test_total_games']}",
                f"{result['new_test_win_rate']:.4f}",
                f"{result['new_test_ci_lo']:.4f}",
                f"{result['new_test_ci_hi']:.4f}",
                f"{result['new_test_std_seeds']:.4f}",
                f"{result['train_test_gap']:.4f}" if result.get('train_test_gap') is not None else ''
            ])
    
    print(f"Saved standardized test results to: {csv_filename}")
    
    # Also create a summary report
    report_filename = results_dir / f'standardized_test_summary_{timestamp}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STANDARDIZED TEST EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Protocol:\n")
        f.write(f"  - 6x6: seeds={test_seeds_6x6}, {games_per_seed} games/seed\n")
        f.write(f"  - 8x8: seeds={test_seeds_8x8}, {games_per_seed} games/seed\n")
        f.write(f"  - Opponent: random baseline (fixed set per seed)\n\n")
        
        # Group by board size
        results_6x6 = [r for r in re_evaluated if r['board_size'] == 6]
        results_8x8 = [r for r in re_evaluated if r['board_size'] == 8]
        
        if results_6x6:
            f.write("=" * 80 + "\n")
            f.write("6x6 STRATEGIES\n")
            f.write("=" * 80 + "\n\n")
            
            for result in sorted(results_6x6, key=lambda x: x['new_test_win_rate'], reverse=True):
                f.write(f"File: {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                f.write(f"  Training Fitness: {result['fitness']:.4f}\n")
                f.write(f"  Test: {result['new_test_wins']}/{result['new_test_total_games']} -> {result['new_test_win_rate']:.1%} [95% CI: {result['new_test_ci_lo']:.1%}-{result['new_test_ci_hi']:.1%}]\n")
                f.write(f"  Variance across seeds (std): {result['new_test_std_seeds']:.3f}\n")
                if result.get('train_test_gap') is not None:
                    f.write(f"  Train-test gap: {result['train_test_gap']:.1%}\n")
                f.write("\n")
            
            # Statistics
            win_rates = [r['new_test_win_rate'] for r in results_6x6]
            f.write(f"6x6 Statistics:\n")
            f.write(f"  Mean Win Rate: {np.mean(win_rates):.1%}\n")
            f.write(f"  Std Dev: {np.std(win_rates):.1%}\n")
            f.write(f"  Min: {np.min(win_rates):.1%}, Max: {np.max(win_rates):.1%}\n")
            f.write(f"  Number of runs: {len(results_6x6)}\n\n")
        
        if results_8x8:
            f.write("=" * 80 + "\n")
            f.write("8x8 STRATEGIES\n")
            f.write("=" * 80 + "\n\n")
            
            for result in sorted(results_8x8, key=lambda x: x['new_test_win_rate'], reverse=True):
                f.write(f"File: {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                f.write(f"  Training Fitness: {result['fitness']:.4f}\n")
                f.write(f"  Test: {result['new_test_wins']}/{result['new_test_total_games']} -> {result['new_test_win_rate']:.1%} [95% CI: {result['new_test_ci_lo']:.1%}-{result['new_test_ci_hi']:.1%}]\n")
                f.write(f"  Variance across seeds (std): {result['new_test_std_seeds']:.3f}\n")
                if result.get('train_test_gap') is not None:
                    f.write(f"  Train-test gap: {result['train_test_gap']:.1%}\n")
                f.write("\n")
            
            # Statistics
            win_rates = [r['new_test_win_rate'] for r in results_8x8]
            f.write(f"8x8 Statistics:\n")
            f.write(f"  Mean Win Rate: {np.mean(win_rates):.1%}\n")
            f.write(f"  Std Dev: {np.std(win_rates):.1%}\n")
            f.write(f"  Min: {np.min(win_rates):.1%}, Max: {np.max(win_rates):.1%}\n")
            f.write(f"  Number of runs: {len(results_8x8)}\n\n")
    
    print(f"Saved summary report to: {report_filename}")
    print()
    print("=" * 80)
    print("RE-EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Re-evaluated {len(re_evaluated)} strategies")
    print(f"Results saved to:")
    print(f"  - {csv_filename}")
    print(f"  - {report_filename}")


if __name__ == '__main__':
    main()
