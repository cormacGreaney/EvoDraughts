import argparse
import os
import sys
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from deap import base, creator, tools

# Ensure project root and grape are on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

grape_path = os.path.join(PROJECT_ROOT, "grape")
if grape_path not in sys.path:
    sys.path.insert(0, grape_path)

import grape  # type: ignore
import algorithms  # type: ignore

from engine_eval.minimax_eval import _play_single_game

from evolve_draughts_8x8 import (
    BNF_GRAMMAR,
    BOARD_SIZE,
    MAX_MOVES,
    strategy_from_phenotype as eval_strategy_from_phenotype,
)


def minimax_fitness_eval(
    individual,
    games_per_eval: int,
    depth: int,
) -> Tuple[float]:
    """
    Fitness for evolving strategies specifically against a minimax-based engine.
    Lower fitness is better (minimisation).
    """
    if getattr(individual, "invalid", False):
        return np.nan,

    try:
        phenotype = individual.phenotype
        strategy = eval_strategy_from_phenotype(phenotype)

        wins = 0
        draws = 0
        losses = 0

        for i in range(games_per_eval):
            as_white = (i % 2 == 0)
            result = _play_single_game(
                strategy,
                strategy_as_white=as_white,
                depth=depth,
                max_moves=MAX_MOVES,
            )
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

        total_games = wins + draws + losses
        if total_games == 0:
            return 1.0,

        win_rate = (wins + 0.5 * draws) / total_games
        fitness = 1.0 - win_rate

        phenotype_str = phenotype
        complexity_tokens = [
            "if_",
            "add",
            "sub",
            "mul",
            "pdiv",
            "max_",
            "min_",
            "greater_than_or_equal",
            "less_than_or_equal",
        ]
        complexity = sum(1 for c in complexity_tokens if c in phenotype_str)
        x_uses = phenotype_str.count("x[")

        # Penalise very simple expressions that only read a single feature
        # and use none of the core combinators.
        try:
            from engine_eval import minimax_config as mm_cfg
            penalty_value = mm_cfg.COMPLEXITY_PENALTY_MINIMAX_8x8
        except Exception:
            penalty_value = 0.30

        if complexity == 0 and 1 <= x_uses <= 2:
            fitness += penalty_value

        return fitness,
    except Exception:
        return 1.0,


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve 8x8 draughts strategies specifically against a minimax-based engine."
    )
    parser.add_argument(
        "--population",
        type=int,
        default=120,
        help="Population size.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations.",
    )
    parser.add_argument(
        "--games-per-eval",
        type=int,
        default=10,
        help="Number of games per fitness evaluation.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Minimax search depth for the engine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    try:
        from engine_eval import minimax_config as mm_cfg

        parser.set_defaults(
            population=mm_cfg.POPULATION_SIZE_MINIMAX_8x8,
            generations=mm_cfg.MAX_GENERATIONS_MINIMAX_8x8,
            games_per_eval=mm_cfg.GAMES_PER_EVAL_MINIMAX_8x8,
            depth=mm_cfg.ENGINE_DEPTH_MINIMAX_8x8,
            seed=mm_cfg.RANDOM_SEED_MINIMAX_8x8,
        )
    except Exception:
        pass

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pop_size = args.population
    n_generations = args.generations
    games_per_eval = args.games_per_eval
    engine_depth = args.depth

    print("=" * 60)
    print("Evolving 8x8 strategies against minimax-based engine")
    print("=" * 60)
    print(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Population Size: {pop_size}")
    print(f"Generations: {n_generations}")
    print(f"Games per Evaluation: {games_per_eval}")
    print(f"Engine depth: {engine_depth}")
    print(f"Max moves per game: {MAX_MOVES}")
    print("=" * 60)
    print()

    # Default GE parameters for this specialised run
    CODON_SIZE = 255
    CODON_CONSUMPTION = "lazy"
    GENOME_REPRESENTATION = "list"
    MAX_GENOME_LENGTH = None
    MAX_INIT_TREE_DEPTH = 13
    MIN_INIT_TREE_DEPTH = 3
    MAX_TREE_DEPTH = 30
    P_CROSSOVER = 0.8
    P_MUTATION = 0.10
    ELITE_SIZE = 1

    try:
        from config import (
            MAX_INIT_TREE_DEPTH_8x8,
            MIN_INIT_TREE_DEPTH_8x8,
            MAX_TREE_DEPTH_8x8,
            P_CROSSOVER as CFG_P_CROSSOVER,
            P_MUTATION_8x8,
            ELITE_SIZE as CFG_ELITE_SIZE,
            CODON_SIZE as CFG_CODON_SIZE,
            CODON_CONSUMPTION as CFG_CODON_CONSUMPTION,
            GENOME_REPRESENTATION as CFG_GENOME_REPRESENTATION,
            MAX_GENOME_LENGTH as CFG_MAX_GENOME_LENGTH,
        )

        MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH_8x8
        MIN_INIT_TREE_DEPTH = MIN_INIT_TREE_DEPTH_8x8
        MAX_TREE_DEPTH = MAX_TREE_DEPTH_8x8
        P_CROSSOVER = CFG_P_CROSSOVER
        P_MUTATION = P_MUTATION_8x8
        ELITE_SIZE = CFG_ELITE_SIZE
        CODON_SIZE = CFG_CODON_SIZE
        CODON_CONSUMPTION = CFG_CODON_CONSUMPTION
        GENOME_REPRESENTATION = CFG_GENOME_REPRESENTATION
        MAX_GENOME_LENGTH = CFG_MAX_GENOME_LENGTH
    except ImportError:
        pass

    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMinEngine"):
        creator.create("FitnessMinEngine", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "EngineIndividual"):
        creator.create("EngineIndividual", grape.Individual, fitness=creator.FitnessMinEngine)

    toolbox.register(
        "populationCreator",
        grape.random_initialisation,
        creator.EngineIndividual,
    )

    def _wrapped_eval(individual, points=None):
        return minimax_fitness_eval(
            individual,
            games_per_eval=games_per_eval,
            depth=engine_depth,
        )

    toolbox.register("evaluate", _wrapped_eval)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    REPORT_ITEMS = [
        "gen",
        "invalid",
        "avg",
        "std",
        "min",
        "max",
        "best_ind_length",
        "avg_length",
        "best_ind_nodes",
        "avg_nodes",
        "avg_depth",
        "avg_used_codons",
        "best_ind_used_codons",
        "selection_time",
        "generation_time",
    ]

    print("Creating initial population...")
    population = toolbox.populationCreator(
        pop_size=pop_size,
        bnf_grammar=BNF_GRAMMAR,
        min_init_genome_length=50,
        max_init_genome_length=150,
        max_init_depth=MAX_INIT_TREE_DEPTH,
        codon_size=CODON_SIZE,
        codon_consumption=CODON_CONSUMPTION,
        genome_representation=GENOME_REPRESENTATION,
    )
    print(f"Created {len(population)} individuals")
    print()

    hof = tools.HallOfFame(10)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    print("Starting evolution against minimax engine...")
    print()
    population, logbook = algorithms.ge_eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=n_generations,
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
        verbose=True,
    )

    print()
    print("=" * 60)
    print("Evolution against minimax engine complete")
    print("=" * 60)

    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    results_dir = Path(PROJECT_ROOT) / "engine_eval" / "minimax_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = results_dir / f"evolution_minimax_8x8_{timestamp}.txt"
    csv_path = results_dir / f"statistics_minimax_8x8_{timestamp}.csv"

    best = hof.items[0] if len(hof.items) > 0 else None
    test_line = ""

    if best is not None:
        print("\nBest Individual (phenotype):")
        print(best.phenotype)
        print(f"Fitness (engine-based): {best.fitness.values[0]:.4f}")

        best_strategy = eval_strategy_from_phenotype(best.phenotype)
        test_games = 20
        wins = draws = losses = 0
        for i in range(test_games):
            as_white = (i % 2 == 0)
            result = _play_single_game(
                best_strategy,
                strategy_as_white=as_white,
                depth=engine_depth,
                max_moves=MAX_MOVES,
            )
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

        test_line = (
            f"Test vs minimax engine (depth={engine_depth}, games={test_games}): "
            f"W={wins}, D={draws}, L={losses}"
        )

        print(f"\n{test_line}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EvoDraughts Minimax-Targeted Evolution Results (8x8)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Board Size: {BOARD_SIZE}x{BOARD_SIZE}\n")
        f.write(f"Population Size: {pop_size}\n")
        f.write(f"Generations: {n_generations}\n")
        f.write(f"Games per Evaluation: {games_per_eval}\n")
        f.write(f"Engine depth: {engine_depth}\n")
        f.write(f"Max moves per game: {MAX_MOVES}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write("=" * 60 + "\n\n")

        if best is not None:
            f.write("Best Individual:\n")
            f.write("-" * 60 + "\n")
            f.write(best.phenotype + "\n\n")
            f.write(f"Engine-based Fitness: {best.fitness.values[0]:.6f}\n")
            if test_line:
                f.write(test_line + "\n\n")

        f.write("Generation Statistics:\n")
        f.write("-" * 60 + "\n")
        for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
            f.write(f"Generation {g}: Best={min_f:.6f}, Avg={avg_f:.6f}\n")

    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Best_Fitness", "Avg_Fitness"])
        for g, min_f, avg_f in zip(gen, min_fitness, avg_fitness):
            writer.writerow([g, min_f, avg_f])

    print(f"Results saved to: {txt_path}")
    print(f"Statistics saved to: {csv_path}")


if __name__ == "__main__":
    main()

