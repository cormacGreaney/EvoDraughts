"""
Quick test script to verify the setup works before running full evolution.
"""

import sys
import os

# Add grape directory to path
grape_path = os.path.join(os.path.dirname(__file__), 'grape')
sys.path.insert(0, grape_path)

import grape
import algorithms
from draughts_game import DraughtsBoard, play_game, random_strategy
import random

# Test with 6x6 board
BOARD_SIZE = 6
from draughts_functions import add, sub
import numpy as np

print("Testing setup...")
print("=" * 60)

# Test 1: Import GRAPE
print("[OK] GRAPE imported successfully")

# Test 2: Load grammar
try:
    GRAMMAR_FILE = 'grammars/draughts.bnf'
    BNF_GRAMMAR = grape.Grammar(GRAMMAR_FILE)
    print("[OK] Grammar loaded successfully")
except Exception as e:
    print(f"[FAIL] Grammar loading failed: {e}")
    sys.exit(1)

# Test 3: Create a board
try:
    board = DraughtsBoard(board_size=BOARD_SIZE)
    moves = board.get_valid_moves(1)
    print(f"[OK] Draughts board ({BOARD_SIZE}x{BOARD_SIZE}) created - {len(moves)} valid moves for player 1")
except Exception as e:
    print(f"[FAIL] Board creation failed: {e}")
    sys.exit(1)

# Test 4: Create a simple individual
try:
    from deap import creator, base
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)
    
    # Create a small test population
    toolbox = base.Toolbox()
    toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
    
    population = toolbox.populationCreator(
        pop_size=5,
        bnf_grammar=BNF_GRAMMAR,
        min_init_genome_length=30,
        max_init_genome_length=50,
        max_init_depth=10,
        codon_size=255,
        codon_consumption='lazy',
        genome_representation='list'
    )
    print(f"[OK] Created test population of {len(population)} individuals")
    
    # Check if individuals have phenotypes
    valid_count = sum(1 for ind in population if not ind.invalid)
    print(f"[OK] {valid_count}/{len(population)} individuals have valid phenotypes")
    
    if valid_count > 0:
        # Show an example phenotype
        for ind in population:
            if not ind.invalid:
                print(f"  Example phenotype: {ind.phenotype[:100]}...")
                break
    
except Exception as e:
    print(f"[FAIL] Population creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Play a simple game
try:
    def simple_strategy(features, valid_moves):
        return 0  # Always pick first move
    
    result = play_game(simple_strategy, random_strategy, max_moves=10, board_size=BOARD_SIZE)
    print(f"[OK] Test game completed - Result: {result}")
except Exception as e:
    print(f"[FAIL] Game play failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("All tests passed! Setup is ready for evolution.")
print("=" * 60)

