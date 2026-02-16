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

# Standardized test seeds
TEST_SEED_6x6 = 999
TEST_SEED_8x8 = 888
TEST_GAMES = 20  # Number of test games per strategy


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
    
    # Extract fitness
    fitness_match = re.search(r'Fitness: ([\d.]+)', content)
    
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
        'old_test_wins': int(old_test_match.group(1)) if old_test_match else None,
        'old_test_total': int(old_test_match.group(2)) if old_test_match else None,
    }


def evaluate_strategy(phenotype, board_size, test_seed, test_games):
    """
    Evaluate a strategy using standardized test protocol.
    
    Returns: (wins, draws, losses, win_rate)
    """
    strategy = strategy_from_phenotype(phenotype, board_size)
    
    # Set fixed random seed for standardized evaluation
    random.seed(test_seed)
    np.random.seed(test_seed)
    
    wins = 0
    draws = 0
    losses = 0
    
    for i in range(test_games):
        result = play_game(strategy, random_strategy, max_moves=200 if board_size == 8 else 100, board_size=board_size)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    
    win_rate = wins / test_games if test_games > 0 else 0.0
    return wins, draws, losses, win_rate


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
    
    # Re-evaluate each strategy
    re_evaluated = []
    for i, result in enumerate(parsed_results, 1):
        print(f"[{i}/{len(parsed_results)}] Re-evaluating {result['file']}...", end=' ')
        
        board_size = result['board_size']
        test_seed = TEST_SEED_6x6 if board_size == 6 else TEST_SEED_8x8
        
        try:
            wins, draws, losses, win_rate = evaluate_strategy(
                result['strategy'],
                board_size,
                test_seed,
                TEST_GAMES
            )
            
            result['new_test_wins'] = wins
            result['new_test_draws'] = draws
            result['new_test_losses'] = losses
            result['new_test_win_rate'] = win_rate
            result['test_seed'] = test_seed
            
            re_evaluated.append(result)
            print(f"OK {wins}/{TEST_GAMES} wins ({win_rate:.1%})")
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
            'Training Fitness', 'Old Test (Wins/Total)', 'Old Test Win Rate',
            'New Test (Wins/Draws/Losses)', 'New Test Win Rate', 'Test Seed',
            'Change in Win Rate'
        ])
        
        for result in re_evaluated:
            old_rate = (result['old_test_wins'] / result['old_test_total']) if result['old_test_total'] else None
            change = (result['new_test_win_rate'] - old_rate) if old_rate is not None else None
            
            writer.writerow([
                result['file'],
                result['date'],
                f"{result['board_size']}x{result['board_size']}",
                result['population'],
                result['generations'],
                result['games_per_eval'],
                result['coevolution'],
                result['strategy'],
                result['fitness'],
                f"{result['old_test_wins']}/{result['old_test_total']}" if result['old_test_total'] else 'N/A',
                f"{old_rate:.1%}" if old_rate is not None else 'N/A',
                f"{result['new_test_wins']}/{result['new_test_draws']}/{result['new_test_losses']}",
                f"{result['new_test_win_rate']:.1%}",
                result['test_seed'],
                f"{change:+.1%}" if change is not None else 'N/A'
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
        f.write(f"  - 6x6 strategies: seed={TEST_SEED_6x6}, {TEST_GAMES} games\n")
        f.write(f"  - 8x8 strategies: seed={TEST_SEED_8x8}, {TEST_GAMES} games\n")
        f.write(f"  - All strategies of same board size tested against identical random opponent sequence\n\n")
        
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
                if result['old_test_total']:
                    f.write(f"  Old Test: {result['old_test_wins']}/{result['old_test_total']} ({result['old_test_wins']/result['old_test_total']:.1%})\n")
                f.write(f"  New Test: {result['new_test_wins']}/{TEST_GAMES} ({result['new_test_win_rate']:.1%})\n")
                f.write(f"  Draws: {result['new_test_draws']}, Losses: {result['new_test_losses']}\n")
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
                if result['old_test_total']:
                    f.write(f"  Old Test: {result['old_test_wins']}/{result['old_test_total']} ({result['old_test_wins']/result['old_test_total']:.1%})\n")
                f.write(f"  New Test: {result['new_test_wins']}/{TEST_GAMES} ({result['new_test_win_rate']:.1%})\n")
                f.write(f"  Draws: {result['new_test_draws']}, Losses: {result['new_test_losses']}\n")
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
