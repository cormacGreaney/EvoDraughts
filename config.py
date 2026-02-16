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

# Iteration 2: Improved 6x6 board with move evaluation
ITERATION_2 = {
    'board_size': 6,
    'description': 'Improved 6x6 draughts with position evaluation',
    'max_moves': 100,
    'grammar_file': 'grammars/draughts2.bnf'
}

# Iteration 3: Standard 8x8 board with move evaluation
ITERATION_3 = {
    'board_size': 8,
    'description': 'Standard 8x8 draughts with position evaluation',
    'max_moves': 200,
    'grammar_file': 'grammars/draughts_8x8.bnf'
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

# Improved version settings (for evolve_draughts2.py)
# - Increased parameters for better results with move evaluation
# - Move evaluation makes each game much slower, but we need more evolution
# - Estimated time: 40-60 hours total
POPULATION_SIZE_V2 = 300
MAX_GENERATIONS_V2 = 50
N_GAMES_PER_EVAL_V2 = 25

# Alternative options (uncomment to use):
# QUICK OPTION (1-2 hours): POPULATION_SIZE = 75, MAX_GENERATIONS = 5, N_GAMES_PER_EVAL = 8
# THOROUGH OPTION (6-10 hours): POPULATION_SIZE = 150, MAX_GENERATIONS = 15, N_GAMES_PER_EVAL = 15

# Genetic operators
P_CROSSOVER = 0.8
P_MUTATION = 0.05  # Increased from 0.01 to prevent premature convergence and encourage exploration
ELITE_SIZE = 1
HALLOFFAME_SIZE = 1

# Improved version mutation rate (for evolve_draughts2.py)
# Increased to prevent premature convergence and encourage diversity
P_MUTATION_V2 = 0.10  # Increased from 0.05 to 0.10 for better exploration

# Improved version hall of fame (for evolve_draughts2.py)
HALLOFFAME_SIZE_V2 = 10

# ============================================================================
# 8x8 VERSION SETTINGS - RUN 1 (First 8x8 run - 2025-12-16)
# ============================================================================
# - 8x8 games are slower due to more positions and longer games
# - Using similar parameters to V2 which worked well for 6x6
# - Estimated time: 60-100 hours total (longer due to game complexity)
# - RESULTS: Severe overfitting (85% training, 10% test), premature convergence
POPULATION_SIZE_8x8_RUN1 = 300
MAX_GENERATIONS_8x8_RUN1 = 50
N_GAMES_PER_EVAL_8x8_RUN1 = 20
P_MUTATION_8x8_RUN1 = 0.10
HALLOFFAME_SIZE_8x8_RUN1 = 10

# ============================================================================
# 8x8 VERSION SETTINGS - RUN 2 (Second 8x8 run - 2025-12-16)
# ============================================================================
# - Increased population (300->400) for more diversity
# - Increased generations (50->75) to prevent premature convergence
# - Increased mutation (0.10->0.15) to maintain diversity
# - Estimated time: 120-200 hours (2-2.5x longer than Run 1)
# - Changes: More random opponents (60%->40% co-evolved), stronger complexity penalty
POPULATION_SIZE_8x8 = 400  # Increased from 300 for Run 1
MAX_GENERATIONS_8x8 = 75   # Increased from 50 for Run 1
N_GAMES_PER_EVAL_8x8 = 20  # Same as Run 1

# 8x8 version mutation rate (for evolve_draughts_8x8.py)
# Increased from 0.10 to prevent premature convergence
P_MUTATION_8x8 = 0.15  # Increased from 0.10 for Run 1

# 8x8 version hall of fame (for evolve_draughts_8x8.py)
HALLOFFAME_SIZE_8x8 = 10  # Same as Run 1

# 8x8 version initialization (for evolve_draughts_8x8.py)
MAX_INIT_TREE_DEPTH_8x8 = 13
MIN_INIT_TREE_DEPTH_8x8 = 3
MAX_TREE_DEPTH_8x8 = 30

# Initialization
MAX_INIT_TREE_DEPTH = 8
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 15

# Improved version initialization (for evolve_draughts2.py)
MAX_INIT_TREE_DEPTH_V2 = 13
MIN_INIT_TREE_DEPTH_V2 = 3
MAX_TREE_DEPTH_V2 = 30
CODON_SIZE = 255
CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None

# Evaluation
N_GAMES_PER_EVAL = 20  # Number of games played per strategy evaluation (more = better fitness estimates)

# Random seed for reproducibility (changed to get different evolutionary path)
# Run 2 used 123, Run 3 uses 456 for different evolutionary path
RANDOM_SEED = 456

# Co-evolution settings (for evolve_draughts2.py)
USE_COEVOLUTION = True
COEVOLUTION_OPPONENTS = 5
USE_SELF_PLAY = False

# 8x8 co-evolution opponent ratio (for evolve_draughts_8x8.py)
# Run 1: 60% co-evolved, 40% random (caused overfitting)
# Run 2: 40% co-evolved, 60% random (to improve generalization)
COEVOLUTION_RATIO_8x8_RUN1 = 0.6  # 60% co-evolved opponents
COEVOLUTION_RATIO_8x8 = 0.4       # 40% co-evolved, 60% random (Run 2)

# 8x8 complexity penalty (for evolve_draughts_8x8.py)
# Run 1: 0.15 (not strong enough - simple strategies still won)
# Run 2: 0.20 (increased to encourage more complex strategies)
COMPLEXITY_PENALTY_8x8_RUN1 = 0.15
COMPLEXITY_PENALTY_8x8 = 0.20  # Increased from 0.15 for Run 2

