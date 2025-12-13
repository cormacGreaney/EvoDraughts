"""
Configuration file for draughts evolution.
Easily switch between different game complexities for multiple iterations.
"""

# ============================================================================
# GAME COMPLEXITY ITERATIONS
# ============================================================================

# Iteration 1: Simplified 6x6 board
ITERATION_1 = {
    'board_size': 6,
    'description': 'Simplified 6x6 draughts',
    'max_moves': 100,
    'grammar_file': 'grammars/draughts.bnf'
}

# Iteration 2: Standard 8x8 board (for future use)
ITERATION_2 = {
    'board_size': 8,
    'description': 'Standard 8x8 draughts',
    'max_moves': 200,
    'grammar_file': 'grammars/draughts.bnf'  # Will need to update grammar for 8x8
}

# Current iteration configuration
CURRENT_ITERATION = ITERATION_1

# ============================================================================
# EVOLUTION PARAMETERS
# ============================================================================

# Population and generation settings
# - 100 generations gives extensive evolution time
# - 200 population provides excellent diversity and prevents premature convergence
# - 20 games per eval gives highly reliable fitness estimates
# - Estimated time: 4-7 hours (perfect for overnight run)
POPULATION_SIZE = 200
MAX_GENERATIONS = 100

# Alternative options (uncomment to use):
# QUICK OPTION (1-2 hours): POPULATION_SIZE = 75, MAX_GENERATIONS = 5, N_GAMES_PER_EVAL = 8
# THOROUGH OPTION (6-10 hours): POPULATION_SIZE = 150, MAX_GENERATIONS = 15, N_GAMES_PER_EVAL = 15

# Genetic operators
P_CROSSOVER = 0.8
P_MUTATION = 0.05  # Increased from 0.01 to prevent premature convergence and encourage exploration
ELITE_SIZE = 1
HALLOFFAME_SIZE = 1

# Initialization
MAX_INIT_TREE_DEPTH = 8
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 15
CODON_SIZE = 255
CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None

# Evaluation
N_GAMES_PER_EVAL = 20  # Number of games played per strategy evaluation (more = better fitness estimates)

# Random seed for reproducibility (changed to get different evolutionary path)
RANDOM_SEED = 123

