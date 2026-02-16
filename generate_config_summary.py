"""
Generate experimental configuration summary table.
This addresses the supervisor's concern about reporting clarity by explicitly documenting
all experimental runs, configurations, and parameters.
"""

import os
import re
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_result_file(filepath):
    """Parse an evolution result file and extract configuration information."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata
    date_match = re.search(r'Date: (.+)', content)
    board_size_match = re.search(r'Board Size: (\d+)x(\d+)', content)
    pop_size_match = re.search(r'Population Size: (\d+)', content)
    gens_match = re.search(r'Generations: (\d+)', content)
    games_per_eval_match = re.search(r'Games per Evaluation: (\d+)', content)
    coevol_match = re.search(r'Co-evolution: (True|False)', content)
    coevol_opponents_match = re.search(r'Co-evolution Opponents: (\d+)', content)
    
    # Extract best strategy
    strategy_match = re.search(r'Best Individual:.*?\n-+\n(.+?)\n\n', content, re.DOTALL)
    
    # Extract fitness and test results
    fitness_match = re.search(r'Fitness: ([\d.]+)', content)
    test_match = re.search(r'Test Results.*?(\d+)/(\d+) wins', content)
    
    # Try to extract random seed from config (if mentioned in file)
    seed_match = re.search(r'Random Seed: (\d+)', content)
    
    return {
        'file': os.path.basename(filepath),
        'filepath': str(filepath),
        'date': date_match.group(1) if date_match else 'Unknown',
        'board_size': int(board_size_match.group(1)) if board_size_match else None,
        'population': int(pop_size_match.group(1)) if pop_size_match else None,
        'generations': int(gens_match.group(1)) if gens_match else None,
        'games_per_eval': int(games_per_eval_match.group(1)) if games_per_eval_match else None,
        'coevolution': coevol_match.group(1) == 'True' if coevol_match else False,
        'coevolution_opponents': int(coevol_opponents_match.group(1)) if coevol_opponents_match else None,
        'strategy': strategy_match.group(1).strip() if strategy_match else None,
        'fitness': float(fitness_match.group(1)) if fitness_match else None,
        'test_wins': int(test_match.group(1)) if test_match else None,
        'test_total': int(test_match.group(2)) if test_match else None,
        'random_seed': int(seed_match.group(1)) if seed_match else None,
    }


def get_config_key(result):
    """Generate a configuration key to group similar runs."""
    key_parts = [
        f"{result['board_size']}x{result['board_size']}",
        f"pop{result['population']}",
        f"gen{result['generations']}",
        f"games{result['games_per_eval']}",
        "coevol" if result['coevolution'] else "random",
    ]
    if result['coevolution'] and result['coevolution_opponents']:
        key_parts.append(f"opp{result['coevolution_opponents']}")
    
    return "_".join(key_parts)


def generate_summary_table():
    """Generate the experimental configuration summary."""
    results_dir = Path('results')
    
    # Find all evolution result files
    result_files = list(results_dir.glob('evolution_*.txt'))
    
    if not result_files:
        print("No evolution result files found in results/ directory")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Parse all results
    all_results = []
    for filepath in result_files:
        try:
            result = parse_result_file(filepath)
            if result['board_size']:  # Only include if we got valid board size
                all_results.append(result)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not all_results:
        print("No valid results to summarize")
        return
    
    # Group by configuration
    config_groups = defaultdict(list)
    for result in all_results:
        config_key = get_config_key(result)
        config_groups[config_key].append(result)
    
    # Generate markdown summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'experimental_config_summary_{timestamp}.md'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Experimental Configuration Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This document provides a comprehensive summary of all experimental runs conducted for the EvoDraughts project.\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Runs**: {len(all_results)}\n")
        f.write(f"- **6x6 Runs**: {len([r for r in all_results if r['board_size'] == 6])}\n")
        f.write(f"- **8x8 Runs**: {len([r for r in all_results if r['board_size'] == 8])}\n")
        f.write(f"- **Unique Configurations**: {len(config_groups)}\n\n")
        
        # Configuration summary table
        f.write("## Configuration Summary\n\n")
        f.write("| Config ID | Board | Pop | Gens | Games/Eval | Co-Evol | Co-Evol Opps | # Runs | Description |\n")
        f.write("|-----------|-------|-----|------|-----------|---------|--------------|--------|-------------|\n")
        
        config_id = 1
        config_mapping = {}
        
        for config_key, runs in sorted(config_groups.items()):
            first_run = runs[0]
            board = f"{first_run['board_size']}x{first_run['board_size']}"
            pop = first_run['population']
            gens = first_run['generations']
            games = first_run['games_per_eval']
            coevol = "Yes" if first_run['coevolution'] else "No"
            coevol_opps = first_run['coevolution_opponents'] if first_run['coevolution'] else "-"
            num_runs = len(runs)
            
            # Generate description
            if first_run['board_size'] == 6:
                if first_run['coevolution']:
                    desc = "6x6 with co-evolution"
                else:
                    desc = "6x6 baseline (random opponents)"
            else:
                if first_run['coevolution']:
                    desc = "8x8 with co-evolution"
                else:
                    desc = "8x8 baseline (random opponents)"
            
            config_id_str = f"Config_{config_id}"
            config_mapping[config_key] = config_id_str
            
            f.write(f"| {config_id_str} | {board} | {pop} | {gens} | {games} | {coevol} | {coevol_opps} | {num_runs} | {desc} |\n")
            config_id += 1
        
        f.write("\n")
        
        # Test evaluation protocol
        f.write("## Test Evaluation Protocol\n\n")
        f.write("| Board Size | Random Seed | Test Games | Opponent Type | Notes |\n")
        f.write("|------------|-------------|------------|---------------|-------|\n")
        f.write("| 6x6 | 999 | 20 | Random | Standardized for all 6x6 strategies |\n")
        f.write("| 8x8 | 888 | 20 | Random | Standardized for all 8x8 strategies |\n")
        f.write("\n")
        f.write("**Note**: All strategies of the same board size are tested against the same random opponent sequence (using fixed random seed) to ensure fair comparison.\n\n")
        
        # Detailed run information
        f.write("## Detailed Run Information\n\n")
        f.write("| Run ID | Config | Result File | Date | Best Strategy | Training Fitness | Test Win Rate |\n")
        f.write("|--------|-------|-------------|------|---------------|------------------|---------------|\n")
        
        for result in sorted(all_results, key=lambda x: (x['board_size'], x['date'])):
            config_key = get_config_key(result)
            config_id_str = config_mapping.get(config_key, "Unknown")
            
            # Truncate strategy if too long
            strategy = result['strategy'] or "N/A"
            if len(strategy) > 50:
                strategy = strategy[:47] + "..."
            
            test_rate = "N/A"
            if result['test_wins'] is not None and result['test_total']:
                test_rate = f"{100 * result['test_wins'] / result['test_total']:.1f}%"
            
            fitness_str = f"{result['fitness']:.4f}" if result['fitness'] else "N/A"
            
            # Extract run ID from filename
            run_id = result['file'].replace('evolution_', '').replace('.txt', '')
            
            f.write(f"| {run_id} | {config_id_str} | {result['file']} | {result['date']} | `{strategy}` | {fitness_str} | {test_rate} |\n")
        
        f.write("\n")
        
        # Configuration statistics
        f.write("## Configuration Statistics\n\n")
        for config_key, runs in sorted(config_groups.items()):
            config_id_str = config_mapping.get(config_key, "Unknown")
            first_run = runs[0]
            
            f.write(f"### {config_id_str}: {first_run['board_size']}x{first_run['board_size']} Board\n\n")
            f.write(f"- **Population Size**: {first_run['population']}\n")
            f.write(f"- **Generations**: {first_run['generations']}\n")
            f.write(f"- **Games per Evaluation**: {first_run['games_per_eval']}\n")
            f.write(f"- **Co-evolution**: {'Yes' if first_run['coevolution'] else 'No'}\n")
            if first_run['coevolution']:
                f.write(f"- **Co-evolution Opponents**: {first_run['coevolution_opponents']}\n")
            f.write(f"- **Number of Runs**: {len(runs)}\n")
            
            # Calculate statistics
            if any(r['test_wins'] for r in runs if r['test_wins'] is not None):
                test_rates = [r['test_wins'] / r['test_total'] for r in runs if r['test_wins'] is not None and r['test_total']]
                if test_rates:
                    f.write(f"- **Test Win Rate**: Mean={np.mean(test_rates):.1%}, Std={np.std(test_rates):.1%}, Range=[{np.min(test_rates):.1%}, {np.max(test_rates):.1%}]\n")
            
            f.write(f"- **Runs**: {', '.join([r['file'] for r in runs])}\n")
            f.write("\n")
        
        # Notes
        f.write("## Notes\n\n")
        f.write("- All test evaluations use standardized random seeds for fair comparison\n")
        f.write("- Training fitness is calculated during evolution (may use different opponents for co-evolution runs)\n")
        f.write("- Test win rate is against standardized random opponent\n")
        f.write("- Some runs may have missing data if result files are incomplete\n\n")
    
    print(f"Configuration summary saved to: {output_file}")
    
    # Also generate CSV version
    csv_file = results_dir / f'experimental_config_summary_{timestamp}.csv'
    with open(csv_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Run ID', 'Config ID', 'Board Size', 'Population', 'Generations',
            'Games per Eval', 'Co-evolution', 'Co-evolution Opponents',
            'Result File', 'Date', 'Best Strategy', 'Training Fitness',
            'Test Wins', 'Test Total', 'Test Win Rate'
        ])
        
        for result in sorted(all_results, key=lambda x: (x['board_size'], x['date'])):
            config_key = get_config_key(result)
            config_id_str = config_mapping.get(config_key, "Unknown")
            
            run_id = result['file'].replace('evolution_', '').replace('.txt', '')
            test_rate = None
            if result['test_wins'] is not None and result['test_total']:
                test_rate = result['test_wins'] / result['test_total']
            
            writer.writerow([
                run_id,
                config_id_str,
                f"{result['board_size']}x{result['board_size']}",
                result['population'],
                result['generations'],
                result['games_per_eval'],
                result['coevolution'],
                result['coevolution_opponents'] or '',
                result['file'],
                result['date'],
                result['strategy'] or '',
                result['fitness'] or '',
                result['test_wins'] or '',
                result['test_total'] or '',
                f"{test_rate:.4f}" if test_rate is not None else ''
            ])
    
    print(f"CSV version saved to: {csv_file}")
    print()
    print("=" * 80)
    print("CONFIGURATION SUMMARY GENERATED")
    print("=" * 80)
    print(f"Total runs documented: {len(all_results)}")
    print(f"Unique configurations: {len(config_groups)}")
    print(f"Output files:")
    print(f"  - {output_file}")
    print(f"  - {csv_file}")


if __name__ == '__main__':
    generate_summary_table()
