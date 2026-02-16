# 8x8 Experimental Runs Tracker

This document tracks the execution of planned 8x8 runs as specified in `planned_8x8_runs.md`.

## Run Status Overview

| Run # | Configuration | Status | Started | Completed | Result File | Notes |
|-------|--------------|--------|---------|-----------|-------------|-------|
| 1 | Config_5 (Run 1) | Complete | 2025-12-16 | 2025-12-16 | evolution_8x8_20251216_222007.txt | Overfitting issue |
| 2 | Config_6 (Run 2) | Complete | 2025-12-18 | 2025-12-18 | evolution_8x8_20251218_002100.txt | Good generalization |
| 3 | Config_6 Replication | Complete | 2026-02-14 | 2026-02-16 | evolution_8x8_20260216_052949.txt | First replication - 75% test (better than Run 2!) |
| 4 | Config_6 Replication |  Ready to Start | - | - | - | Second replication run |
| 5 | Config_6 Replication | Pending | - | - | - | Third replication run |
| 6 | Co-evolution 30% | Pending | - | - | - | 30% co-evolved, 70% random |
| 7 | Co-evolution 50% | Pending | - | - | - | 50% co-evolved, 50% random |
| 8 | Co-evolution 20% | Pending | - | - | - | 20% co-evolved, 80% random |
| 9 | Mutation 0.10 | Pending | - | - | - | Lower mutation rate |
| 10 | Mutation 0.20 | Pending | - | - | - | Higher mutation rate |

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

### Test Evaluation Protocol
- **Random Seed**: 888 (for all 8x8 runs)
- **Test Games**: 20
- **Opponent Type**: Random baseline

---

## Notes
- All runs use standardized test evaluation (seed 888, 20 games)
- Each run takes approximately 120-200 hours
- Results will be analyzed as they complete
