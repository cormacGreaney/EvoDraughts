"""
Script to load and play against evolved draughts strategies.
Loads strategies from results files and lets you play against them.
"""

import sys
import os
import re

# Add grape directory to path
grape_path = os.path.join(os.path.dirname(__file__), 'grape')
sys.path.insert(0, grape_path)

import numpy as np
from draughts_game import DraughtsBoard, play_game
from draughts_functions import add, sub, mul, pdiv, max_, min_, greater_than_or_equal, less_than_or_equal, if_, neg, abs_


def strategy_from_phenotype(phenotype, board_size=6):
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
            # For 6x6: 36 board + 11 features = 47, for 8x8: 64 + 11 = 75
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
            
            # Convert result to move index
            if isinstance(result, (np.ndarray, list)):
                if len(result) > 0:
                    result = float(result[0] if isinstance(result, np.ndarray) else result[0])
                else:
                    result = 0.0
            elif isinstance(result, (bool, np.bool_)):
                result = 1.0 if result else 0.0
            else:
                result = float(result)
            
            # Ensure result is a number and convert to move index
            try:
                move_idx = int(abs(result)) % len(valid_moves)
            except (ValueError, TypeError, OverflowError):
                move_idx = 0
            
            return move_idx
        except Exception as e:
            # If evaluation fails, return random move
            import random
            return random.randint(0, len(valid_moves) - 1) if valid_moves else 0
    
    return strategy


def load_strategy_from_file(results_file):
    """
    Load a strategy phenotype from a results file.
    Returns: (phenotype_string, board_size, fitness, metadata_dict)
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract board size
    board_size_match = re.search(r'Board Size: (\d+)x\d+', content)
    board_size = int(board_size_match.group(1)) if board_size_match else 6
    
    # Extract phenotype (between "Best Individual:" and next section)
    phenotype_match = re.search(r'Best Individual:\s*-+\s*(.+?)\s+Fitness:', content, re.DOTALL)
    if not phenotype_match:
        raise ValueError("Could not find phenotype in results file")
    
    phenotype = phenotype_match.group(1).strip()
    
    # Extract fitness
    fitness_match = re.search(r'Fitness: ([\d.]+)', content)
    fitness = float(fitness_match.group(1)) if fitness_match else None
    
    # Extract other metadata
    metadata = {
        'board_size': board_size,
        'fitness': fitness,
        'file': results_file
    }
    
    depth_match = re.search(r'Depth: (\d+)', content)
    if depth_match:
        metadata['depth'] = int(depth_match.group(1))
    
    return phenotype, board_size, fitness, metadata


def human_strategy(features, valid_moves):
    """Simple strategy that returns first move (placeholder for future human input)."""
    return 0


def random_strategy(features, valid_moves):
    """Random baseline strategy."""
    import random
    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def play_against_strategy(results_file, num_games=10, opponent='random'):
    """
    Play games against an evolved strategy.
    
    Args:
        results_file: Path to the results file containing the evolved strategy
        num_games: Number of games to play
        opponent: 'random' or 'human' (currently only 'random' works)
    """
    print("=" * 60)
    print("Loading Evolved Strategy")
    print("=" * 60)
    
    try:
        phenotype, board_size, fitness, metadata = load_strategy_from_file(results_file)
        print(f"Loaded strategy from: {os.path.basename(results_file)}")
        print(f"Board Size: {board_size}x{board_size}")
        if fitness is not None:
            print(f"Strategy Fitness: {fitness:.4f} (lower is better)")
        if 'depth' in metadata:
            print(f"Strategy Depth: {metadata['depth']}")
        print()
        
        # Wrap phenotype for display
        import textwrap
        print("Strategy Phenotype:")
        print("-" * 60)
        print("\n".join(textwrap.wrap(phenotype, 80)))
        print()
        
        # Create strategy function
        evolved_strategy = strategy_from_phenotype(phenotype, board_size)
        
        # Choose opponent
        if opponent == 'random':
            opponent_strategy = random_strategy
            opponent_name = "Random"
        else:
            raise ValueError("Only 'random' opponent is currently supported")
        
        print("=" * 60)
        print(f"Playing {num_games} Games: Evolved Strategy vs {opponent_name}")
        print("=" * 60)
        
        wins_evolved = 0
        wins_opponent = 0
        draws = 0
        
        max_moves = 100 if board_size == 6 else 200
        
        for game_num in range(1, num_games + 1):
            # Play as player 1 (evolved strategy)
            result = play_game(evolved_strategy, opponent_strategy, 
                             max_moves=max_moves, board_size=board_size)
            
            if result == 1:
                wins_evolved += 1
                outcome = "Evolved Strategy wins!"
            elif result == -1:
                wins_opponent += 1
                outcome = f"{opponent_name} wins!"
            else:
                draws += 1
                outcome = "Draw"
            
            print(f"Game {game_num}/{num_games}: {outcome}")
        
        print()
        print("=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Evolved Strategy Wins: {wins_evolved}/{num_games} ({wins_evolved/num_games:.1%})")
        print(f"{opponent_name} Wins: {wins_opponent}/{num_games} ({wins_opponent/num_games:.1%})")
        print(f"Draws: {draws}/{num_games} ({draws/num_games:.1%})")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def list_available_strategies(results_dir="results"):
    """List all available evolved strategies in the results directory."""
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist.")
        print("Run the evolution first to generate strategies.")
        return []
    
    files = [f for f in os.listdir(results_dir) if f.startswith("evolution_") and f.endswith(".txt")]
    files.sort(reverse=True)  # Most recent first
    
    if not files:
        print(f"No strategy files found in '{results_dir}' directory.")
        return []
    
    print(f"Found {len(files)} strategy file(s) in '{results_dir}':")
    print()
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(results_dir, filename)
        print(f"{i}. {filename}")
        
        # Try to extract metadata
        try:
            phenotype, board_size, fitness, metadata = load_strategy_from_file(filepath)
            if fitness is not None:
                print(f"   Fitness: {fitness:.4f}, Board: {board_size}x{board_size}")
        except:
            pass
        print()
    
    return files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play against evolved draughts strategies')
    parser.add_argument('--file', type=str, help='Path to results file containing the strategy')
    parser.add_argument('--list', action='store_true', help='List all available strategies')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play (default: 10)')
    parser.add_argument('--opponent', type=str, default='random', choices=['random'], 
                       help='Opponent type (default: random)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_strategies()
    elif args.file:
        play_against_strategy(args.file, num_games=args.games, opponent=args.opponent)
    else:
        print("Usage:")
        print("  python play_against_evolved.py --list                    # List available strategies")
        print("  python play_against_evolved.py --file results/evolution_TIMESTAMP.txt  # Play against a strategy")
        print()
        print("Examples:")
        print("  python play_against_evolved.py --list")
        print("  python play_against_evolved.py --file results/evolution_20241220_143022.txt --games 20")
        print()
        list_available_strategies()

