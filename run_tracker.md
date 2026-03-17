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
| 6 | Co-evol 30% (Condition A) replicate 1 | Complete | 2026-02-21 | 2026-02-23 | evolution_8x8_20260223_045614.txt | 2001 |
| 7 | Co-evol 30% replicate 2 | Complete | 2026-02-23 | 2026-02-25 | evolution_8x8_20260225_114105.txt | 2002 |
| 8 | Co-evol 30% replicate 3 | Complete | 2026-02-25 | 2026-02-27 | evolution_8x8_20260227_011809.txt | 2003 |
| 9 | Co-evol 50% (Condition B) replicate 1 | Complete | 2026-02-27 | 2026-03-02 | evolution_8x8_20260302_115155.txt | 2011 |
| 10 | Co-evol 50% replicate 2 | Complete | 2026-03-02 | 2026-03-07 | evolution_8x8_20260307_093010.txt | 2012 |
| 11 | Co-evol 50% replicate 3 | Complete | 2026-03-07 | 2026-03-09 | evolution_8x8_20260309_155326.txt | 2013 |
| 12 | Mutation 0.10 (Condition C) replicate 1 | Complete | 2026-03-09 | 2026-03-11 | evolution_8x8_20260311_043951.txt | 3011 |
| 13 | Mutation 0.10 replicate 2 | Complete | 2026-03-11 | 2026-03-12 | evolution_8x8_20260312_214315.txt | 3012 |
| 14 | Mutation 0.10 replicate 3 | Complete | 2026-03-12 | 2026-03-16 | evolution_8x8_20260316_010459.txt | 3013 |
| 15 | Mutation 0.20 (Condition D) replicate 1 | In Progress | 2026-03-16 | - | - | 3021 |
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

### Run 6 (Condition A: Co-evol 30%) replicate 1 - COMPLETE
- **Date Started**: 2026-02-21
- **Date Completed**: 2026-02-23 04:56:14
- **Configuration**: Condition A — 30% co-evolved, 70% random; Population 400, Gen 75, Games 20, Mutation 0.15, Complexity 0.20
  - Evolution seed: 2001
- **Result File**: evolution_8x8_20260223_045614.txt
- **Best Strategy**: `mul(add(abs_(10.0), x[26]), x[23])`
- **Training Fitness**: 0.0750 (92.5% win rate)
- **Test (5 seeds × 30 games)**: 114/150 → 76.0% [95% CI 69.2%–82.8%], variance across seeds 0.061
- **Notes**: First replicate of Condition A. Strong test performance.

### Run 7 (Condition A: Co-evol 30%) replicate 2 - COMPLETE
- **Date Started**: 2026-02-23
- **Date Completed**: 2026-02-25 11:41:05
- **Configuration**: Same as Run 6 (30% co-evolved, 70% random). Evolution seed: 2002
- **Result File**: evolution_8x8_20260225_114105.txt
- **Best Strategy**: `1.0`
- **Training Fitness**: 0.1000 (90.0% win rate)
- **Test (5 seeds × 30 games)**: 104/150 → 69.3% [95% CI 62.0%–76.7%], variance across seeds 0.013
- **Notes**: Second replicate of Condition A.

### Run 8 (Condition A: Co-evol 30%) replicate 3 - COMPLETE
- **Date Started**: 2026-02-25
- **Date Completed**: 2026-02-27 01:18:09
- **Configuration**: Same as Run 6 (30% co-evolved, 70% random). Evolution seed: 2003
- **Result File**: evolution_8x8_20260227_011809.txt
- **Best Strategy**: `add(if_(greater_than_or_equal(x[35], x[64]), x[44], x[44]), x[37])`
- **Training Fitness**: 0.1000 (90.0% win rate)
- **Test (5 seeds × 30 games)**: 100/150 → 66.7% [95% CI 59.1%–74.2%], variance across seeds 0.063
- **Notes**: Third replicate of Condition A. Condition A (30% co-evol) complete.

### Run 9 (Condition B: Co-evol 50%) replicate 1 - COMPLETE
- **Date Started**: 2026-02-27
- **Date Completed**: 2026-03-02 11:51:55
- **Configuration**: Condition B — 50% co-evolved, 50% random; Population 400, Gen 75, Games 20, Mutation 0.15, Complexity 0.20. Evolution seed: 2011
- **Result File**: evolution_8x8_20260302_115155.txt
- **Best Strategy**: `add(x[24], x[68])`
- **Training Fitness**: 0.0750 (92.5% win rate)
- **Test (5 seeds × 30 games)**: 108/150 → 72.0% [95% CI 64.8%–79.2%], variance across seeds 0.027
- **Notes**: First replicate of Condition B. Solid generalization with moderate variance across seeds.

### Run 10 (Condition B: Co-evol 50%) replicate 2 - COMPLETE
- **Date Started**: 2026-03-02
- **Date Completed**: 2026-03-07 09:30:10
- **Configuration**: Condition B — 50% co-evolved, 50% random; Population 400, Gen 75, Games 20, Mutation 0.15, Complexity 0.20. Evolution seed: 2012
- **Result File**: evolution_8x8_20260307_093010.txt
- **Best Strategy**: `sub(x[53], neg(x[70]))`
- **Training Fitness**: 0.0500 (95.0% win rate)
- **Test (5 seeds × 30 games)**: 110/150 → 73.3% [95% CI 66.3%–80.4%], variance across seeds 0.092
- **Notes**: Second replicate of Condition B.

### Run 11 (Condition B: Co-evol 50%) replicate 3 - COMPLETE
- **Date Started**: 2026-03-07
- **Date Completed**: 2026-03-09 15:53:26
- **Configuration**: Condition B — 50% co-evolved, 50% random; Population 400, Gen 75, Games 20, Mutation 0.15, Complexity 0.20. Evolution seed: 2013
- **Result File**: evolution_8x8_20260309_155326.txt
- **Best Strategy**: `add(x[68], sub(x[24], x[46]))`
- **Training Fitness**: 0.0500 (95.0% win rate)
- **Test (5 seeds × 30 games)**: 122/150 → 81.3% [95% CI 75.1%–87.6%], variance across seeds 0.062
- **Notes**: Third replicate of Condition B. Best standardized test performance so far.

### Run 12 (Condition C: Mutation 0.10) replicate 1 - COMPLETE
- **Date Started**: 2026-03-09
- **Date Completed**: 2026-03-11 04:39:51
- **Configuration**: Condition C — mutation 0.10; Population 400, Gen 75, Games 20, Co-evol 40%, Complexity 0.20. Evolution seed: 3011
- **Result File**: evolution_8x8_20260311_043951.txt
- **Best Strategy**: `add(add(x[40], x[39]), x[19])`
- **Training Fitness**: 0.0500 (95.0% win rate)
- **Test (5 seeds × 30 games)**: 117/150 → 78.0% [95% CI 71.4%–84.6%], variance across seeds 0.050
- **Notes**: First replicate of Condition C (lower mutation). Very strong test performance, slightly below Run 11 but better than most co-evolution runs.

### Run 13 (Condition C: Mutation 0.10) replicate 2 - COMPLETE
- **Date Started**: 2026-03-11
- **Date Completed**: 2026-03-12 21:43:15
- **Configuration**: Condition C — mutation 0.10; Population 400, Gen 75, Games 20, Co-evol 40%, Complexity 0.20. Evolution seed: 3012
- **Result File**: evolution_8x8_20260312_214315.txt
- **Best Strategy**: `sub(x[70], x[46])`
- **Training Fitness**: 0.0750 (92.5% win rate)
- **Test (5 seeds × 30 games)**: 105/150 → 70.0% [95% CI 62.7%–77.3%], variance across seeds 0.087
- **Notes**: Second replicate of Condition C. Solid performance, slightly below Run 12 and clearly below Run 11.

### Run 14 (Condition C: Mutation 0.10) replicate 3 - COMPLETE
- **Date Started**: 2026-03-12
- **Date Completed**: 2026-03-16 01:04:59
- **Configuration**: Condition C — mutation 0.10; Population 400, Gen 75, Games 20, Co-evol 40%, Complexity 0.20. Evolution seed: 3013
- **Result File**: evolution_8x8_20260316_010459.txt
- **Best Strategy**: `mul(x[72], x[39])`
- **Training Fitness**: 0.0250 (97.5% win rate)
- **Test (5 seeds × 30 games)**: 95/150 → 63.3% [95% CI 55.6%–71.0%], variance across seeds 0.092
- **Notes**: Third replicate of Condition C. Weaker test performance than Runs 12–13 and clearly below the best co-evolution runs.

### Runs 15–17 (Condition D: Mutation 0.20) replicates 1–3
- **Run 15 (IN PROGRESS)**:
  - Evolution seed: 3021. Mutation rate 0.20; co-evolution ratio 40% co-evolved, 60% random (same as Config_6 baseline). Config is set for Run 15; start with: `python evolve_draughts_8x8.py`
- **Runs 16–17 (PENDING)**:
  - Seeds: 3022, 3023. Mutation rate 0.20; rest as Config_6.

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
| 6 | Co-evol 30% rep 1 | 2001 | evolution_8x8_20260223_045614.txt |
| 7 | Co-evol 30% rep 2 | 2002 | evolution_8x8_20260225_114105.txt |
| 8 | Co-evol 30% rep 3 | 2003 | evolution_8x8_20260227_011809.txt |
| 9 | Co-evol 50% rep 1 | 2011 | evolution_8x8_20260302_115155.txt |
| 10 | Co-evol 50% rep 2 | 2012 | evolution_8x8_20260307_093010.txt |
| 11 | Co-evol 50% rep 3 | 2013 | evolution_8x8_20260309_155326.txt |
| 12 | Mutation 0.10 rep 1 | 3011 | evolution_8x8_20260311_043951.txt |
| 13 | Mutation 0.10 rep 2 | 3012 | evolution_8x8_20260312_214315.txt |
| 14 | Mutation 0.10 rep 3 | 3013 | evolution_8x8_20260316_010459.txt |
| 15 | Mutation 0.20 rep 1 | 3021 | - |
| 16 | Mutation 0.20 rep 2 | 3022 | - |
| 17 | Mutation 0.20 rep 3 | 3023 | - |
