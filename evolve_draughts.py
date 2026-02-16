"""
Main script for evolving draughts strategies using GRAPE.
This script runs the grammatical evolution algorithm to evolve game strategies.
"""

import sys
import os

# Add grape directory to path
grape_path = os.path.join(os.path.dirname(__file__), 'grape')
sys.path.insert(0, grape_path)

import grape
import algorithms
from draughts_functions import add, sub, mul, pdiv, max_, min_, greater_than_or_equal, less_than_or_equal, if_, neg, abs_
from draughts_game import DraughtsBoard, play_game
import numpy as np
from deap import creator, base, tools
import random
import warnings

warnings.filterwarnings("ignore")

# Import configuration
try:
    from config import CURRENT_ITERATION, POPULATION_SIZE, MAX_GENERATIONS, P_CROSSOVER, P_MUTATION
    from config import ELITE_SIZE, HALLOFFAME_SIZE, MAX_INIT_TREE_DEPTH, MIN_INIT_TREE_DEPTH
    from config import MAX_TREE_DEPTH, CODON_SIZE, CODON_CONSUMPTION, GENOME_REPRESENTATION
    from config import MAX_GENOME_LENGTH, N_GAMES_PER_EVAL, RANDOM_SEED
    BOARD_SIZE = CURRENT_ITERATION['board_size']
    GRAMMAR_FILE = CURRENT_ITERATION['grammar_file']
    MAX_MOVES = CURRENT_ITERATION['max_moves']
except ImportError:
    # Fallback defaults if config.py doesn't exist
    BOARD_SIZE = 6
    GRAMMAR_FILE = 'grammars/draughts.bnf'
    MAX_MOVES = 100
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 1
    P_CROSSOVER = 0.8
    P_MUTATION = 0.01
    ELITE_SIZE = 1
    HALLOFFAME_SIZE = 1
    MAX_INIT_TREE_DEPTH = 8
    MIN_INIT_TREE_DEPTH = 3
    MAX_TREE_DEPTH = 15
    CODON_SIZE = 255
    CODON_CONSUMPTION = 'lazy'
    GENOME_REPRESENTATION = 'list'
    MAX_GENOME_LENGTH = None
    N_GAMES_PER_EVAL = 5
    RANDOM_SEED = 42

# Set random seed
random.seed(RANDOM_SEED)

# Load grammar
BNF_GRAMMAR = grape.Grammar(GRAMMAR_FILE)


def random_strategy(features, valid_moves):
    """Random baseline strategy for comparison."""
    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def strategy_from_phenotype(phenotype):
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
            expected_features = BOARD_SIZE * BOARD_SIZE + 11
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
            from draughts_functions import add, sub, mul, pdiv, max_, min_, greater_than_or_equal, less_than_or_equal, if_, neg, abs_
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
            return random.randint(0, len(valid_moves) - 1) if valid_moves else 0
    
    return strategy


def fitness_eval(individual, points=None):
    """
    Evaluate fitness of a strategy by playing games.
    Lower fitness is better (we're minimizing).
    points parameter is kept for compatibility with GRAPE but not used.
    """
    if individual.invalid:
        return np.nan,
    
    try:
        # Create strategy from phenotype
        strategy = strategy_from_phenotype(individual.phenotype)
        
        # Play games against random opponent
        wins = 0
        draws = 0
        losses = 0
        
        for _ in range(N_GAMES_PER_EVAL):
            # Play as player 1
            result = play_game(strategy, random_strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            
            # Play as player 2
            result = play_game(random_strategy, strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == -1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
        
        # Fitness is the loss rate (we want to minimize losses)
        total_games = wins + draws + losses
        fitness = losses / total_games if total_games > 0 else 1.0
        
        # Invert so wins contribute negatively to fitness
        # We want: more wins = lower fitness
        fitness = 1.0 - (wins / total_games) if total_games > 0 else 1.0
        
        # Add small complexity bonus to encourage more sophisticated strategies
        # Penalize overly simple strategies (just a single feature lookup)
        # Count nodes in phenotype (rough estimate: count function calls and operators)
        phenotype_str = individual.phenotype
        complexity = len([c for c in ['if_', 'add', 'sub', 'mul', 'pdiv', 'max_', 'min_', 
                                      'greater_than_or_equal', 'less_than_or_equal'] 
                          if c in phenotype_str])
        
        # If strategy is too simple (just x[i]), add small penalty
        # This encourages evolution to find more complex solutions
        if complexity == 0 and 'x[' in phenotype_str and phenotype_str.count('x[') == 1:
            # Very simple strategy (just x[i]), add small penalty
            fitness += 0.05  # Small penalty to encourage complexity
        
        return fitness,
        
    except Exception as e:
        # If strategy evaluation fails, return worst fitness
        return 1.0,


# Create DEAP toolbox
toolbox = base.Toolbox()

# Define a single objective, minimizing fitness strategy
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

# Use random_initialisation for now (sensible_initialisation has a float issue in some Python versions)
toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
toolbox.register("evaluate", fitness_eval)

# Tournament selection (increased tournament size for stronger selection pressure)
toolbox.register("select", tools.selTournament, tournsize=7)

# Single-point crossover
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
                'best_ind_length', 'avg_length',
                'best_ind_nodes', 'avg_nodes',
                'avg_depth',
                'avg_used_codons', 'best_ind_used_codons',
                'selection_time', 'generation_time']


def main():
    """Run the evolutionary algorithm."""
    print("=" * 60)
    print("Evolving Draughts Strategies with GRAPE")
    print("=" * 60)
    try:
        from config import CURRENT_ITERATION
        print(f"Iteration: {CURRENT_ITERATION['description']}")
    except:
        pass
    print(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Games per Evaluation: {N_GAMES_PER_EVAL}")
    print("=" * 60)
    print()
    
    # Create initial population (generation 0)
    print("Creating initial population...")
    population = toolbox.populationCreator(
        pop_size=POPULATION_SIZE,
        bnf_grammar=BNF_GRAMMAR,
        min_init_genome_length=30,
        max_init_genome_length=100,
        max_init_depth=MAX_INIT_TREE_DEPTH,
        codon_size=CODON_SIZE,
        codon_consumption=CODON_CONSUMPTION,
        genome_representation=GENOME_REPRESENTATION
    )
    print(f"Created {len(population)} individuals")
    print()
    
    # Define hall of fame
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    
    # Prepare statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # Run evolutionary algorithm
    print("Starting evolution...")
    print()
    population, logbook = algorithms.ge_eaSimpleWithElitism(
        population, toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        elite_size=ELITE_SIZE,
        bnf_grammar=BNF_GRAMMAR,
        codon_size=CODON_SIZE,
        max_tree_depth=MAX_TREE_DEPTH,
        max_genome_length=MAX_GENOME_LENGTH,
        points_train=None,  # Not used in our fitness function
        codon_consumption=CODON_CONSUMPTION,
        report_items=REPORT_ITEMS,
        genome_representation=GENOME_REPRESENTATION,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # Display results
    print()
    print("=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    
    if len(hof.items) > 0:
        best = hof.items[0]
        print("\nBest Individual:")
        print("-" * 60)
        import textwrap
        print("\n".join(textwrap.wrap(best.phenotype, 80)))
        print(f"\nFitness: {best.fitness.values[0]:.4f}")
        print(f"Depth: {best.depth}")
        print(f"Genome Length: {len(best.genome)}")
        print(f"Used Codons: {best.used_codons}/{len(best.genome)} ({best.used_codons/len(best.genome):.2%})")
        
        # Test the best strategy
        print("\n" + "=" * 60)
        print("Testing Best Strategy")
        print("=" * 60)
        best_strategy = strategy_from_phenotype(best.phenotype)
        
        test_wins = 0
        test_games = 10
        for i in range(test_games):
            result = play_game(best_strategy, random_strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == 1:
                test_wins += 1
        
        print(f"Wins against random: {test_wins}/{test_games} ({test_wins/test_games:.1%})")
    
    # Extract and display statistics
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    
    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
        print(f"Generation {g}: Best={min_f:.4f}, Avg={avg_f:.4f}")
    
    # Save results to file
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    import os
    import csv
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created {results_dir}/ directory")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/evolution_{timestamp}.txt"
    
    # Save detailed results to text file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EvoDraughts Evolution Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}\n")
        f.write(f"Population Size: {POPULATION_SIZE}\n")
        f.write(f"Generations: {MAX_GENERATIONS}\n")
        f.write(f"Games per Evaluation: {N_GAMES_PER_EVAL}\n")
        f.write("=" * 60 + "\n\n")
        
        if len(hof.items) > 0:
            best = hof.items[0]
            f.write("Best Individual:\n")
            f.write("-" * 60 + "\n")
            f.write(best.phenotype + "\n\n")
            f.write(f"Fitness: {best.fitness.values[0]:.4f}\n")
            f.write(f"Depth: {best.depth}\n")
            f.write(f"Genome Length: {len(best.genome)}\n")
            f.write(f"Used Codons: {best.used_codons}/{len(best.genome)} ({best.used_codons/len(best.genome):.2%})\n\n")
            
            # Test results - using standardized random seed for fair comparison
            # All 6x6 strategies tested with seed 999, all 8x8 with seed 888
            TEST_SEED_6x6 = 999
            TEST_SEED_8x8 = 888
            test_seed = TEST_SEED_6x6 if BOARD_SIZE == 6 else TEST_SEED_8x8
            
            best_strategy = strategy_from_phenotype(best.phenotype)
            test_wins = 0
            test_games = 20  # Increased from 10 for better statistics
            test_draws = 0
            test_losses = 0
            
            # Set fixed random seed for standardized test evaluation
            random.seed(test_seed)
            np.random.seed(test_seed)
            
            for i in range(test_games):
                result = play_game(best_strategy, random_strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
                if result == 1:
                    test_wins += 1
                elif result == 0:
                    test_draws += 1
                else:
                    test_losses += 1
            
            f.write(f"Test Results (vs Random, standardized seed={test_seed}): {test_wins}/{test_games} wins ({test_wins/test_games:.1%})\n")
            f.write(f"  Draws: {test_draws}, Losses: {test_losses}\n")
            f.write(f"  Note: All {BOARD_SIZE}x{BOARD_SIZE} strategies tested against same random opponent sequence\n\n")
        
        f.write("Generation Statistics:\n")
        f.write("-" * 60 + "\n")
        for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
            f.write(f"Generation {g}: Best={min_f:.4f}, Avg={avg_f:.4f}\n")
    
    print(f"Results saved to: {filename}")
    
    # Also save CSV file with statistics
    csv_filename = f"{results_dir}/statistics_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness'])
        for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
            writer.writerow([g, min_f, avg_f])
    
    print(f"Statistics saved to: {csv_filename}")
    print("=" * 60)


if __name__ == "__main__":
    main()

