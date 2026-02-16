"""
Main script for evolving draughts strategies using GRAPE (8x8 version).
This version uses position evaluation and move selection, with co-evolution support.
For 8x8 standard draughts board.
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
    from config import CURRENT_ITERATION, P_CROSSOVER, P_MUTATION_8x8
    from config import ELITE_SIZE, CODON_SIZE, CODON_CONSUMPTION, GENOME_REPRESENTATION
    from config import MAX_GENOME_LENGTH, RANDOM_SEED
    from config import USE_COEVOLUTION, COEVOLUTION_OPPONENTS, USE_SELF_PLAY
    from config import POPULATION_SIZE_8x8, MAX_GENERATIONS_8x8, N_GAMES_PER_EVAL_8x8
    from config import MAX_INIT_TREE_DEPTH_8x8, MIN_INIT_TREE_DEPTH_8x8, MAX_TREE_DEPTH_8x8
    from config import HALLOFFAME_SIZE_8x8
    from config import ITERATION_3
    # Run 2: Import new co-evolution ratio and complexity penalty
    from config import COEVOLUTION_RATIO_8x8, COMPLEXITY_PENALTY_8x8
    P_MUTATION = P_MUTATION_8x8  # Use 8x8 mutation rate
    
    BOARD_SIZE = ITERATION_3['board_size']
    GRAMMAR_FILE = ITERATION_3['grammar_file']
    MAX_MOVES = ITERATION_3['max_moves']
    
    POPULATION_SIZE = POPULATION_SIZE_8x8
    MAX_GENERATIONS = MAX_GENERATIONS_8x8
    N_GAMES_PER_EVAL = N_GAMES_PER_EVAL_8x8
    MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH_8x8
    MIN_INIT_TREE_DEPTH = MIN_INIT_TREE_DEPTH_8x8
    MAX_TREE_DEPTH = MAX_TREE_DEPTH_8x8
except ImportError:
    BOARD_SIZE = 8
    GRAMMAR_FILE = 'grammars/draughts_8x8.bnf'
    MAX_MOVES = 200
    POPULATION_SIZE = 300
    MAX_GENERATIONS = 50
    P_CROSSOVER = 0.8
    P_MUTATION = 0.10
    ELITE_SIZE = 1
    HALLOFFAME_SIZE_8x8 = 10
    MAX_INIT_TREE_DEPTH = 13
    MIN_INIT_TREE_DEPTH = 3
    MAX_TREE_DEPTH = 30
    CODON_SIZE = 255
    CODON_CONSUMPTION = 'lazy'
    GENOME_REPRESENTATION = 'list'
    MAX_GENOME_LENGTH = None
    N_GAMES_PER_EVAL = 20
    RANDOM_SEED = 42
    USE_COEVOLUTION = True
    COEVOLUTION_OPPONENTS = 5
    USE_SELF_PLAY = False
    # Run 2: Default values if config import fails
    COEVOLUTION_RATIO_8x8 = 0.4  # 40% co-evolved, 60% random (Run 2)
    COMPLEXITY_PENALTY_8x8 = 0.20  # Increased from 0.15 (Run 2)

# Set random seed
random.seed(RANDOM_SEED)

# Load grammar
BNF_GRAMMAR = grape.Grammar(GRAMMAR_FILE)

# Global variables for co-evolution
current_population = None
hall_of_fame = None


def random_strategy(features, valid_moves):
    """Random baseline strategy for comparison."""
    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def evaluate_position(phenotype, board):
    """
    Evaluate a board position using the phenotype.
    Returns a single numeric value representing position quality.
    """
    try:
        features = board.get_board_features()
        x = np.array(features, dtype=float)
        
        expected_features = BOARD_SIZE * BOARD_SIZE + 11
        if len(x) < expected_features:
            x = np.pad(x, (0, max(0, expected_features - len(x))), 'constant')
        elif len(x) > expected_features:
            x = x[:expected_features]
        
        local_vars = {
            'x': x,
            'np': np
        }
        
        for i in range(len(x)):
            local_vars[f'x[{i}]'] = x[i]
        
        from draughts_functions import add, sub, mul, pdiv, max_, min_, greater_than_or_equal, less_than_or_equal, if_, neg, abs_
        local_vars.update({
            'add': add, 'sub': sub, 'mul': mul, 'pdiv': pdiv,
            'max_': max_, 'min_': min_,
            'greater_than_or_equal': greater_than_or_equal,
            'less_than_or_equal': less_than_or_equal,
            'if_': if_, 'neg': neg, 'abs_': abs_
        })
        
        result = eval(phenotype, {"__builtins__": {}}, local_vars)
        
        if isinstance(result, (np.ndarray, list)):
            if len(result) > 0:
                result = float(result[0] if isinstance(result, np.ndarray) else result[0])
            else:
                result = 0.0
        elif isinstance(result, (bool, np.bool_)):
            result = 1.0 if result else 0.0
        else:
            result = float(result)
        
        return float(result)
    except Exception:
        return 0.0


def strategy_from_phenotype(phenotype):
    """
    Create a strategy function from a GRAPE phenotype.
    The phenotype is a position evaluation function.
    For each valid move, we evaluate the resulting position and pick the best one.
    """
    def strategy(features, valid_moves):
        if not valid_moves:
            return 0
        
        try:
            board = DraughtsBoard(board_size=BOARD_SIZE)
            board_array = np.array(features[:BOARD_SIZE * BOARD_SIZE], dtype=int).reshape(BOARD_SIZE, BOARD_SIZE)
            board.board = board_array
            
            if len(features) > BOARD_SIZE * BOARD_SIZE + 10:
                current_player_val = int(features[BOARD_SIZE * BOARD_SIZE + 10])
                board.current_player = current_player_val if current_player_val in [1, -1] else 1
            
            player_perspective = board.current_player
            
            best_move_idx = 0
            best_eval = float('-inf')
            
            for move_idx, (from_pos, to_pos) in enumerate(valid_moves):
                test_board = board.copy()
                if test_board.make_move(from_pos, to_pos):
                    eval_score = evaluate_position(phenotype, test_board)
                    if player_perspective == -1:
                        eval_score = -eval_score
                    if eval_score > best_eval:
                        best_eval = eval_score
                        best_move_idx = move_idx
            
            return best_move_idx
        except Exception:
            return random.randint(0, len(valid_moves) - 1) if valid_moves else 0
    
    return strategy


def get_opponent_strategy():
    """
    Get an opponent strategy for co-evolution.
    Returns a strategy from the current population or hall of fame.
    Always includes some random opponents to maintain baseline.
    """
    global current_population, hall_of_fame
    
    if USE_COEVOLUTION and current_population is not None:
        opponents = []
        
        if hall_of_fame is not None and len(hall_of_fame.items) > 0:
            for i in range(min(COEVOLUTION_OPPONENTS, len(hall_of_fame.items))):
                opponents.append(hall_of_fame.items[i])
        
        if current_population is not None:
            sample_size = min(COEVOLUTION_OPPONENTS, len(current_population))
            sampled = random.sample(current_population, sample_size)
            opponents.extend(sampled)
        
        if opponents:
            if random.random() < 0.3:
                return random_strategy
            opponent = random.choice(opponents)
            return strategy_from_phenotype(opponent.phenotype)
    
    return random_strategy


def fitness_eval(individual, points=None):
    """
    Evaluate fitness of a strategy by playing games.
    Lower fitness is better (we're minimizing).
    Uses co-evolution when enabled: plays against mix of random and evolved opponents.
    """
    if individual.invalid:
        return np.nan,
    
    try:
        strategy = strategy_from_phenotype(individual.phenotype)
        
        wins = 0
        draws = 0
        losses = 0
        
        # Use co-evolution if enabled, otherwise use random only
        # Run 1: 60% co-evolved, 40% random (caused overfitting)
        # Run 2: 40% co-evolved, 60% random (to improve generalization)
        for game_idx in range(N_GAMES_PER_EVAL):
            # Determine opponent based on co-evolution setting
            # Run 1: if USE_COEVOLUTION and random.random() < 0.6:  # 60% co-evolved
            # Run 2: Changed to use COEVOLUTION_RATIO_8x8 (0.4 = 40% co-evolved)
            if USE_COEVOLUTION and random.random() < COEVOLUTION_RATIO_8x8:
                opponent_strategy = get_opponent_strategy()
            else:
                opponent_strategy = random_strategy
            
            # Play as player 1
            result = play_game(strategy, opponent_strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            
            # Play as player 2
            result = play_game(opponent_strategy, strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == -1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
        
        total_games = wins + draws + losses
        if total_games == 0:
            return 1.0,
        
        win_rate = wins / total_games
        
        # Base fitness: fewer wins -> higher fitness
        fitness = 1.0 - win_rate
        
        # Stronger complexity penalty to discourage trivial x[i]-only strategies
        phenotype_str = individual.phenotype
        complexity_tokens = ['if_', 'add', 'sub', 'mul', 'pdiv', 'max_', 'min_',
                             'greater_than_or_equal', 'less_than_or_equal']
        complexity = sum(1 for c in complexity_tokens if c in phenotype_str)
        
        # Very simple strategy (just x[i]) gets a stronger penalty
        # Run 1: fitness += 0.15  # Not strong enough - simple strategies still won
        # Run 2: Increased to COMPLEXITY_PENALTY_8x8 (0.20) to encourage more complex strategies
        if complexity == 0 and 'x[' in phenotype_str and phenotype_str.count('x[') == 1:
            fitness += COMPLEXITY_PENALTY_8x8  # Run 2: 0.20 (was 0.15 in Run 1)
        
        return fitness,
        
    except Exception:
        # On any error, give worst fitness
        return 1.0,


# Create DEAP toolbox
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
toolbox.register("evaluate", fitness_eval)

toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", grape.crossover_onepoint)
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
                'best_ind_length', 'avg_length',
                'best_ind_nodes', 'avg_nodes',
                'avg_depth',
                'avg_used_codons', 'best_ind_used_codons',
                'selection_time', 'generation_time']


def main():
    """Run the evolutionary algorithm."""
    global current_population, hall_of_fame
    
    print("=" * 60)
    print("Evolving Draughts Strategies with GRAPE (8x8 Version)")
    print("=" * 60)
    try:
        from config import CURRENT_ITERATION
        print(f"Iteration: {ITERATION_3['description']}")
    except:
        pass
    print(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Max Generations: {MAX_GENERATIONS}")
    print(f"Games per Evaluation: {N_GAMES_PER_EVAL}")
    print(f"Co-evolution: {USE_COEVOLUTION}")
    if USE_COEVOLUTION:
        print(f"Co-evolution Opponents: {COEVOLUTION_OPPONENTS}")
    print("=" * 60)
    print()
    
    print("Creating initial population...")
    population = toolbox.populationCreator(
        pop_size=POPULATION_SIZE,
        bnf_grammar=BNF_GRAMMAR,
        min_init_genome_length=50,
        max_init_genome_length=150,
        max_init_depth=MAX_INIT_TREE_DEPTH,
        codon_size=CODON_SIZE,
        codon_consumption=CODON_CONSUMPTION,
        genome_representation=GENOME_REPRESENTATION
    )
    print(f"Created {len(population)} individuals")
    print()
    
    hof = tools.HallOfFame(HALLOFFAME_SIZE_8x8)
    current_population = population
    hall_of_fame = hof
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
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
        points_train=None,
        codon_consumption=CODON_CONSUMPTION,
        report_items=REPORT_ITEMS,
        genome_representation=GENOME_REPRESENTATION,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    current_population = population
    hall_of_fame = hof
    
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
        
        print("\n" + "=" * 60)
        print("Testing Best Strategy")
        print("=" * 60)
        best_strategy = strategy_from_phenotype(best.phenotype)
        
        test_wins = 0
        test_games = 20
        for i in range(test_games):
            result = play_game(best_strategy, random_strategy, max_moves=MAX_MOVES, board_size=BOARD_SIZE)
            if result == 1:
                test_wins += 1
        
        print(f"Wins against random: {test_wins}/{test_games} ({test_wins/test_games:.1%})")
    
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    
    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
        print(f"Generation {g}: Best={min_f:.4f}, Avg={avg_f:.4f}")
    
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    import os
    import csv
    from datetime import datetime
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created {results_dir}/ directory")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/evolution_8x8_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EvoDraughts Evolution Results (8x8 Version)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}\n")
        f.write(f"Population Size: {POPULATION_SIZE}\n")
        f.write(f"Generations: {MAX_GENERATIONS}\n")
        f.write(f"Games per Evaluation: {N_GAMES_PER_EVAL}\n")
        f.write(f"Co-evolution: {USE_COEVOLUTION}\n")
        if USE_COEVOLUTION:
            f.write(f"Co-evolution Opponents: {COEVOLUTION_OPPONENTS}\n")
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
            test_games = 20
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
    
    csv_filename = f"{results_dir}/statistics_8x8_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness'])
        for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
            writer.writerow([g, min_f, avg_f])
    
    print(f"Statistics saved to: {csv_filename}")
    print("=" * 60)


if __name__ == "__main__":
    main()

