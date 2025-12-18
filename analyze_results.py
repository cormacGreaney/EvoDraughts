"""
Analysis script for evolution results.
Parses result files and generates visualizations.
"""

import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_result_file(filepath):
    """Parse an evolution result file and extract key information."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract metadata
    date_match = re.search(r'Date: (.+)', content)
    board_size_match = re.search(r'Board Size: (\d+)x(\d+)', content)
    pop_size_match = re.search(r'Population Size: (\d+)', content)
    gens_match = re.search(r'Generations: (\d+)', content)
    
    # Extract best strategy
    strategy_match = re.search(r'Best Individual:.*?\n-+\n(.+?)\n\n', content, re.DOTALL)
    
    # Extract fitness and test results
    fitness_match = re.search(r'Fitness: ([\d.]+)', content)
    test_match = re.search(r'Test Results.*?(\d+)/(\d+) wins', content)
    
    # Extract generation statistics
    gen_stats = []
    for line in content.split('\n'):
        if line.startswith('Generation ') and 'Best=' in line:
            gen_match = re.search(r'Generation (\d+): Best=([\d.]+), Avg=([\d.]+)', line)
            if gen_match:
                gen_stats.append({
                    'generation': int(gen_match.group(1)),
                    'best_fitness': float(gen_match.group(2)),
                    'avg_fitness': float(gen_match.group(3))
                })
    
    return {
        'file': os.path.basename(filepath),
        'date': date_match.group(1) if date_match else 'Unknown',
        'board_size': int(board_size_match.group(1)) if board_size_match else None,
        'population': int(pop_size_match.group(1)) if pop_size_match else None,
        'generations': int(gens_match.group(1)) if gens_match else None,
        'strategy': strategy_match.group(1).strip() if strategy_match else None,
        'fitness': float(fitness_match.group(1)) if fitness_match else None,
        'test_wins': int(test_match.group(1)) if test_match else None,
        'test_total': int(test_match.group(2)) if test_match else None,
        'generation_stats': gen_stats
    }


def extract_features_from_strategy(strategy_str):
    """Extract feature indices used in a strategy."""
    # Find all x[index] patterns
    feature_matches = re.findall(r'x\[(\d+)\]', strategy_str)
    return [int(f) for f in feature_matches]


def analyze_strategy_complexity(strategy_str):
    """Analyze the complexity of a strategy."""
    complexity_tokens = ['if_', 'add', 'sub', 'mul', 'pdiv', 'max_', 'min_',
                         'greater_than_or_equal', 'less_than_or_equal']
    
    token_counts = {token: strategy_str.count(token) for token in complexity_tokens}
    total_complexity = sum(token_counts.values())
    
    # Count features used
    features = extract_features_from_strategy(strategy_str)
    unique_features = len(set(features))
    
    return {
        'total_operations': total_complexity,
        'unique_features': unique_features,
        'total_features': len(features),
        'token_counts': token_counts,
        'features_used': features
    }


def plot_evolution_progress(results, output_dir='results/analysis'):
    """Create plots showing evolution progress across runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename without extension for cleaner labels
    def get_short_name(filename):
        # Remove 'evolution_' prefix and '.txt' suffix, keep date
        name = filename.replace('evolution_', '').replace('.txt', '')
        # Extract just the date part if it's long
        if len(name) > 20:
            return name[:17] + '...'
        return name
    
    # Plot 1: Best Fitness Over Generations
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for result in results:
        if result['generation_stats']:
            gens = [s['generation'] for s in result['generation_stats']]
            best = [s['best_fitness'] for s in result['generation_stats']]
            label = f"{result['board_size']}x{result['board_size']} - {get_short_name(result['file'])}"
            ax1.plot(gens, best, label=label, marker='o', markersize=4, linewidth=2)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness (lower is better)', fontsize=12)
    ax1.set_title('Best Fitness Over Generations', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_best_fitness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/1_best_fitness.png")
    plt.close()
    
    # Plot 2: Average Fitness Over Generations
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for result in results:
        if result['generation_stats']:
            gens = [s['generation'] for s in result['generation_stats']]
            avg = [s['avg_fitness'] for s in result['generation_stats']]
            label = f"{result['board_size']}x{result['board_size']} - {get_short_name(result['file'])}"
            ax2.plot(gens, avg, label=label, marker='s', markersize=4, linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Average Fitness', fontsize=12)
    ax2.set_title('Average Fitness Over Generations', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_avg_fitness.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/2_avg_fitness.png")
    plt.close()
    
    # Plot 3: Test Performance Comparison
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    board_sizes = []
    test_performances = []
    labels = []
    for result in results:
        if result['test_wins'] is not None and result['test_total'] is not None:
            board_sizes.append(result['board_size'])
            test_perf = (result['test_wins'] / result['test_total']) * 100
            test_performances.append(test_perf)
            labels.append(f"{result['board_size']}x{result['board_size']}\n{get_short_name(result['file'])}")
    
    if board_sizes:
        bars = ax3.bar(range(len(test_performances)), test_performances, 
                       color=['#2ecc71' if p > 50 else '#e74c3c' for p in test_performances])
        ax3.set_xticks(range(len(test_performances)))
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Test Win Rate (%)', fontsize=12)
        ax3.set_title('Test Performance Comparison', fontsize=14, fontweight='bold')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, perf in zip(bars, test_performances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{perf:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_test_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/3_test_performance.png")
    plt.close()
    
    # Plot 4: Training vs Test Performance
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    training_perfs = []
    test_perfs = []
    labels_tt = []
    for result in results:
        if result['fitness'] is not None and result['test_wins'] is not None:
            training_perf = (1 - result['fitness']) * 100
            test_perf = (result['test_wins'] / result['test_total']) * 100 if result['test_total'] else 0
            training_perfs.append(training_perf)
            test_perfs.append(test_perf)
            labels_tt.append(f"{result['board_size']}x{result['board_size']}\n{get_short_name(result['file'])}")
    
    if training_perfs:
        x = np.arange(len(training_perfs))
        width = 0.35
        ax4.bar(x - width/2, training_perfs, width, label='Training', color='#3498db', alpha=0.8)
        ax4.bar(x + width/2, test_perfs, width, label='Test', color='#e67e22', alpha=0.8)
        ax4.set_ylabel('Win Rate (%)', fontsize=12)
        ax4.set_title('Training vs Test Performance', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels_tt, fontsize=10)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_training_vs_test.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/4_training_vs_test.png")
    plt.close()


def generate_report(results, output_dir='results/analysis'):
    """Generate text report summarizing evolution results."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVODRAUGHTS - EVOLUTION RESULTS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Runs Analyzed: {len(results)}\n\n")
        
        # Best performers
        f.write("BEST PERFORMERS\n")
        f.write("-" * 80 + "\n")
        
        best_6x6 = None
        best_8x8 = None
        
        for result in results:
            if result['test_wins'] is not None and result['test_total'] is not None:
                test_perf = result['test_wins'] / result['test_total']
                if result['board_size'] == 6:
                    if best_6x6 is None or test_perf > (best_6x6['test_wins'] / best_6x6['test_total']):
                        best_6x6 = result
                elif result['board_size'] == 8:
                    if best_8x8 is None or test_perf > (best_8x8['test_wins'] / best_8x8['test_total']):
                        best_8x8 = result
        
        if best_6x6:
            f.write(f"\nBest 6x6 Strategy:\n")
            f.write(f"  File: {best_6x6['file']}\n")
            f.write(f"  Strategy: {best_6x6['strategy']}\n")
            f.write(f"  Training Fitness: {best_6x6['fitness']:.4f} ({100*(1-best_6x6['fitness']):.1f}% win rate)\n")
            f.write(f"  Test Performance: {best_6x6['test_wins']}/{best_6x6['test_total']} ({100*best_6x6['test_wins']/best_6x6['test_total']:.1f}%)\n")
            complexity = analyze_strategy_complexity(best_6x6['strategy'])
            f.write(f"  Complexity: {complexity['total_operations']} operations, {complexity['unique_features']} unique features\n")
            f.write(f"  Features Used: {complexity['features_used']}\n")
        
        if best_8x8:
            f.write(f"\nBest 8x8 Strategy:\n")
            f.write(f"  File: {best_8x8['file']}\n")
            f.write(f"  Strategy: {best_8x8['strategy']}\n")
            f.write(f"  Training Fitness: {best_8x8['fitness']:.4f} ({100*(1-best_8x8['fitness']):.1f}% win rate)\n")
            f.write(f"  Test Performance: {best_8x8['test_wins']}/{best_8x8['test_total']} ({100*best_8x8['test_wins']/best_8x8['test_total']:.1f}%)\n")
            complexity = analyze_strategy_complexity(best_8x8['strategy'])
            f.write(f"  Complexity: {complexity['total_operations']} operations, {complexity['unique_features']} unique features\n")
            f.write(f"  Features Used: {complexity['features_used']}\n")
        
        # Strategy comparison
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("STRATEGY COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            if result['strategy']:
                f.write(f"{result['board_size']}x{result['board_size']} - {result['file']}\n")
                f.write(f"  Strategy: {result['strategy']}\n")
                complexity = analyze_strategy_complexity(result['strategy'])
                f.write(f"  Operations: {complexity['total_operations']}\n")
                f.write(f"  Unique Features: {complexity['unique_features']}\n")
                f.write(f"  Token Breakdown: {complexity['token_counts']}\n")
                if result['test_wins'] is not None:
                    f.write(f"  Test: {result['test_wins']}/{result['test_total']} ({100*result['test_wins']/result['test_total']:.1f}%)\n")
                f.write("\n")
    
    print(f"Saved analysis report to {report_path}")


def main():
    """Main analysis function."""
    results_dir = Path('results')
    
    # Find all evolution result files
    result_files = list(results_dir.glob('evolution_*.txt'))
    
    if not result_files:
        print("No evolution result files found in results/ directory")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Parse all results
    results = []
    for filepath in result_files:
        try:
            result = parse_result_file(filepath)
            results.append(result)
            print(f"Parsed: {result['file']}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    if not results:
        print("No valid results to analyze")
        return
    
    # Create analysis directory
    analysis_dir = 'results/analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_evolution_progress(results, analysis_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, analysis_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check results/analysis/ for:")
    print("  - 1_best_fitness.png (best fitness over generations)")
    print("  - 2_avg_fitness.png (average fitness over generations)")
    print("  - 3_test_performance.png (test performance comparison)")
    print("  - 4_training_vs_test.png (training vs test comparison)")
    print("  - analysis_report.txt (detailed text report)")
    print("=" * 80)


if __name__ == '__main__':
    main()

