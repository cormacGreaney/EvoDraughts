# 8x8 Experimental Runs Tracker

This document tracks the execution of planned 8x8 runs as specified in `planned_8x8_runs.md`.

## Run Status Overview

Matches `planned_8x8_runs.md`: Config_6 baseline (runs 3–5), then **two** co-evolution settings (30%, 50%) with 2–3 replicates each, then **two** mutation settings (0.10, 0.20) with 2–3 replicates each.

| Run # | Condition | Status | Started | Completed | Result File | Evolution seed |
|-------|-----------|--------|---------|-----------|-------------|----------------|
| 1 | Config_5 | Complete | 2025-12-16 | 2025-12-16 | evolution_8x8_20251216_222007.txt | not recorded |
| 2 | Config_6 | Complete | 2025-12-18 | 2025-12-18 | evolution_8x8_20251218_002100.txt | 123 |
| 3 | Config_6 replicate 1 | Complete | 2026-02-14 | 2026-02-16 | evolution_8x8_20260216_052949.txt | 456 |
| 4 | Config_6 replicate 2 | Complete | 2026-02-16 | 2026-02-18 | evolution_8x8_20260218_003143.txt | 789 |
| 5 | Config_6 replicate 3 | Complete | 2026-02-18 | 2026-02-19 | evolution_8x8_20260219_203646.txt | 1011 |
| 6 | Co-evol 30% (Condition A) replicate 1 | In Progress | 2026-02-21 | - | - | 2001 |
| 7 | Co-evol 30% replicate 2 | Pending | - | - | - | 2002 |
| 8 | Co-evol 30% replicate 3 | Pending | - | - | - | 2003 |
| 9 | Co-evol 50% (Condition B) replicate 1 | Pending | - | - | - | 2011 |
| 10 | Co-evol 50% replicate 2 | Pending | - | - | - | 2012 |
| 11 | Co-evol 50% replicate 3 | Pending | - | - | - | 2013 |
| 12 | Mutation 0.10 (Condition C) replicate 1 | Pending | - | - | - | 3011 |
| 13 | Mutation 0.10 replicate 2 | Pending | - | - | - | 3012 |
| 14 | Mutation 0.10 replicate 3 | Pending | - | - | - | 3013 |
| 15 | Mutation 0.20 (Condition D) replicate 1 | Pending | - | - | - | 3021 |
| 16 | Mutation 0.20 replicate 2 | Pending | - | - | - | 3022 |
| 17 | Mutation 0.20 replicate 3 | Pending | - | - | - | 3023 |

## Run Details

### Run 1 (Config_5) - COMPLETE
- **Date**: 2025-12-16
- **Configuration**: Population 300, Generations 50, Games/Eval 20, Co-evolution 60%, Mutation 0.10
- **Result File**: evolution_8x8_20251216_222007.txt
- **Training Fitness**: 0.1500 (85% win rate)
- **Standardized Test**: 12/20 wins (60.0%)
- **Notes**: Severe overfitting - 85% training but only 10% on original test (60% on standardized)

### Run 2 (Config_6) - COMPLETE
- **Date**: 2025-12-18
- **Configuration**: Population 400, Generations 75, Games/Eval 20, Co-evolution 40%, Mutation 0.15, Complexity 0.20
- **Result File**: evolution_8x8_20251218_002100.txt
- **Training Fitness**: 0.1750 (82.5% win rate)
- **Standardized Test**: 14/20 wins (70.0%)
- **Notes**: Good generalization, better alignment between training and test

### Run 3 (Config_6 Replication) - COMPLETE
- **Date Started**: 2026-02-14
- **Date Completed**: 2026-02-16 05:29:49
- **Configuration**: Same as Run 2
  - Population: 400
  - Generations: 75
  - Games per evaluation: 20
  - Co-evolution ratio: 40% co-evolved, 60% random
  - Mutation rate: 0.15
  - Complexity penalty: 0.20
  - Evolution random seed: 456 (different from Run 2's 123)
  - Test evaluation seed: 888 (standardized, same as all 8x8 runs)
- **Result File**: evolution_8x8_20260216_052949.txt
- **Best Strategy**: `sub(add(if_(less_than_or_equal(5.0, 2.0), -0.1, add(x[19], 0.0)), x[11]), -0.5)`
- **Training Fitness**: 0.1250 (87.5% win rate)
- **Standardized Test**: 15/20 wins (75.0%)
- **Notes**: First replication of Run 2 configuration. **Better than Run 2!** Run 2 had 70% test performance, Run 3 achieved 75%. Good generalization - training (87.5%) and test (75%) are well-aligned.

### Run 4 (Config_6 Replication) - COMPLETE
- **Date Started**: 2026-02-16
- **Date Completed**: 2026-02-18 00:31:43
- **Configuration**: Same as Run 2 (Population 400, Gen 75, Games 20, Co-evol 40%, Mutation 0.15, Complexity 0.20)
  - Evolution seed: 789 | Test seed: 888
- **Result File**: evolution_8x8_20260218_003143.txt
- **Best Strategy**: `add(if_(greater_than_or_equal(mul(x[49], 10.0), x[57]), x[39], 0.5), x[19])`
- **Training Fitness**: 0.0750 (92.5% win rate)
- **Standardized Test**: 17/20 wins (85.0%)
- **Notes**: Second replication. **Best test result so far** (85%). Good generalization.

### Run 5 (Config_6 Replication) - COMPLETE
- **Date Started**: 2026-02-18
- **Date Completed**: 2026-02-19 20:36:46
- **Configuration**: Same as Run 2 (Population 400, Gen 75, Games 20, Co-evol 40%, Mutation 0.15, Complexity 0.20)
  - Evolution seed: 1011 | Test evaluation: see Re-evaluation below
- **Result File**: evolution_8x8_20260219_203646.txt
- **Best Strategy**: `mul(x[73], x[24])`
- **Training Fitness**: 0.0500 (95.0% win rate)
- **In-file test (old protocol)**: 18/20 wins (90.0%)
- **Re-evaluation (new protocol, 5 seeds × 30 games)**: 94/150 → 62.7% [95% CI 54.9%–70.4%], train–test gap 0.32
- **Notes**: Third replication. Strong training; re-eval with multiple seeds gives more conservative test estimate.

### Run 6 (Condition A: Co-evol 30%) replicate 1 - IN PROGRESS
- **Date Started**: 2026-02-21
- **Configuration**: Condition A in plan — same as Config_6 except co-evolution ratio
  - Co-evolution ratio: **30%** co-evolved, **70%** random
  - Population: 400, Generations: 75, Games per evaluation: 20, Mutation 0.15, Complexity 0.20
  - Evolution seed: 2001
- **Result File**: [when complete]
- **Notes**: First of 2–3 replicates for Condition A. Compare vs Config_6 (runs 3–5) and Condition B (50%, runs 9–11).

### Runs 7–8 (Condition A: Co-evol 30%) replicates 2–3 - PENDING
- **Seeds**: 2002, 2003. Same config as Run 6.

### Runs 9–11 (Condition B: Co-evol 50%) replicates 1–3 - PENDING
- **Seeds**: 2011, 2012, 2013. Co-evolution ratio 50% / 50% random; rest as Config_6.

### Runs 12–14 (Condition C: Mutation 0.10) replicates 1–3 - PENDING
- **Seeds**: 3011, 3012, 3013. Mutation rate 0.10; rest as Config_6.

### Runs 15–17 (Condition D: Mutation 0.20) replicates 1–3 - PENDING
- **Seeds**: 3021, 3022, 3023. Mutation rate 0.20; rest as Config_6.

---

## Configuration Reference

### Config_6 (Run 2, 3, 4, 5)
```python
POPULATION_SIZE_8x8 = 400
MAX_GENERATIONS_8x8 = 75
N_GAMES_PER_EVAL_8x8 = 20
COEVOLUTION_RATIO_8x8 = 0.4  # 40% co-evolved, 60% random
P_MUTATION_8x8 = 0.15
COMPLEXITY_PENALTY_8x8 = 0.20
```

---

## Evolution seeds (by run)

| Run # | Condition | Evolution seed | Result file |
|-------|-----------|----------------|-------------|
| 1 | Config_5 | not recorded | evolution_8x8_20251216_222007.txt |
| 2 | Config_6 | 123 | evolution_8x8_20251218_002100.txt |
| 3 | Config_6 rep 1 | 456 | evolution_8x8_20260216_052949.txt |
| 4 | Config_6 rep 2 | 789 | evolution_8x8_20260218_003143.txt |
| 5 | Config_6 rep 3 | 1011 | evolution_8x8_20260219_203646.txt |
| 6 | Co-evol 30% rep 1 | 2001 | (in progress) |
| 7 | Co-evol 30% rep 2 | 2002 | - |
| 8 | Co-evol 30% rep 3 | 2003 | - |
| 9 | Co-evol 50% rep 1 | 2011 | - |
| 10 | Co-evol 50% rep 2 | 2012 | - |
| 11 | Co-evol 50% rep 3 | 2013 | - |
| 12 | Mutation 0.10 rep 1 | 3011 | - |
| 13 | Mutation 0.10 rep 2 | 3012 | - |
| 14 | Mutation 0.10 rep 3 | 3013 | - |
| 15 | Mutation 0.20 rep 1 | 3021 | - |
| 16 | Mutation 0.20 rep 2 | 3022 | - |
| 17 | Mutation 0.20 rep 3 | 3023 | - |
