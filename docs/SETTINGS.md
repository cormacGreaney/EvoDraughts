# Settings - What I've Learned

After running three evolution experiments, here's what I've learned about the settings and what works.

## My Current Settings

```python
POPULATION_SIZE = 200
MAX_GENERATIONS = 100
N_GAMES_PER_EVAL = 20
P_MUTATION = 0.05
P_CROSSOVER = 0.8
Tournament Size = 7
Complexity Penalty = Active
```

These take about 2-3 hours for 100 generations on a 6x6 board.

## Why Each Setting Matters

**Population Size (200)**: Large enough to maintain diversity and prevent premature convergence. Smaller populations (100) converged too quickly to simple solutions.

**Generations (100)**: Enough time to see clear evolution. I observed fitness improve from ~0.45 (generation 0) to ~0.275-0.325 (generation 100). My 10-generation runs were too short.

**Games per Evaluation (20)**: More games = more reliable fitness scores. I tried 10 games and the fitness was noisier. 20 gives stable estimates.

**Mutation Rate (0.05)**: My initial runs with 1% mutation converged too early. 5% keeps evolution exploring while still improving. This was a key finding.

**Tournament Size (7)**: Stronger selection pressure. I increased this from 5 to improve selection of better strategies.

**Complexity Penalty**: Essential! Without it, evolution found trivial solutions like `x[8]`. With it, Run 3 evolved `if_(greater_than_or_equal(...))` - actual conditional logic.

## What I Tried

**Run 1** (10 gen, 100 pop, 10 games): Quick test, showed evolution works. Best: `x[32]`, fitness 0.30, 20% win rate. Too simple.

**Run 2** (100 gen, 200 pop, 20 games): Better evolution but still converged to simple strategies. Best: `x[8]`, fitness 0.275, 30% win rate. Still too simple.

**Run 3** (100 gen, 200 pop, 20 games, complexity penalty): Evolved complex strategies. Best: `if_(greater_than_or_equal(x[0], x[25]), x[19], x[20])`, fitness 0.325, 40% win rate. Much better!

## What Works

- Higher mutation (0.05) prevents premature convergence
- Complexity penalty encourages sophisticated strategies
- Larger populations (200) maintain diversity
- More games per eval (20) gives reliable fitness

## What Doesn't Work

- Too low mutation (0.01) - converges too early
- No complexity pressure - finds trivial solutions
- Too few games per eval (<10) - noisy fitness
- Too small population (<100) - loses diversity quickly

## Progress Observed

- **Win Rate**: 20% → 30% → 40% across the three runs
- **Strategy Complexity**: Simple `x[i]` → Conditional logic with comparisons
- **Fitness**: 0.45 → 0.30 → 0.275-0.325 over runs

The key insight: Higher mutation and complexity penalty are essential. Without them, evolution finds trivial solutions too quickly.

