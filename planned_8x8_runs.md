# Planned 8x8 Experimental Runs

## Current Status

I've completed two 8x8 runs so far. The first run (Config_5) showed severe overfitting - it achieved 85% training performance but only 10% on the standardized test (though the re-evaluation with standardized protocol shows 60%, which is still a significant gap). The second run (Config_6) performed much better, getting 70% on the standardized test with better alignment between training and test performance.

## Planned Runs

### 1. Replication Runs (Runs 3-5)
**Configuration**: Same as Run 2 (Config_6)
- Population: 400
- Generations: 75
- Games per evaluation: 20
- Co-evolution ratio: 40% co-evolved, 60% random
- Mutation rate: 0.15
- Complexity penalty: 0.20

**Rationale**: Run 2 showed decent results with good generalization. I need to replicate this configuration a few times to establish statistical confidence and understand the run-to-run variance. This addresses your feedback about needing multiple runs and explicit variance reporting.

**Expected outcome**: If this configuration is actually robust, I should see consistent performance across the three runs. If there's high variance, that will tell me something important about the stability of the approach.

---

### 2. Co-evolution Ratio Testing (Runs 6-8)
**Configuration**: Same as Run 2, but testing different co-evolution ratios
- Run 6: 30% co-evolved, 70% random
- Run 7: 50% co-evolved, 50% random  
- Run 8: 20% co-evolved, 80% random

**Rationale**: Run 1 used 60% co-evolved and overfit badly. Run 2 used 40% and generalized much better. For now I want to properly explore this parameter to understand where the sweet spot is between maintaining diversity (co-evolution) and ensuring generalization (random baseline). This will help me understand why some runs generalize and others don't.

**Expected outcome**: I expect to see a trade-off - too much co-evolution leads to overfitting, too little might not provide enough evolutionary pressure. This should help find the optimal ratio.

---

### 3. Mutation Rate Variation (Runs 9-10)
**Configuration**: Same as Run 2, but varying mutation rates
- Run 9: Mutation rate 0.10 (lower than Run 2's 0.15)
- Run 10: Mutation rate 0.20 (higher than Run 2's 0.15)

**Rationale**: Run 2 increased mutation from 0.10 to 0.15 to prevent premature convergence. I want to see if this was the right choice or if we can do better. Lower mutation might lead to premature convergence, higher mutation might prevent the population from settling on good solutions.

**Expected outcome**: This will help me understand how mutation rate affects both convergence and generalization. If 0.15 is optimal, these runs should perform worse. If not, I'll have data to adjust.

---

## Notes

- All runs will use the standardized test evaluation (seed 888, 20 test games) that I've now implemented, so results will be directly comparable.

- I'm focusing solely on 8x8 for the moment, since that's the full game scale and where I've been able to see the most interesting challenges like overfitting.

- Each run takes approximately 120-200 hours, so this plan represents a significant computational investment. I'll run them sequentially and analyze results as they complete.

- The main goal is to understand variance, identify what causes overfitting, and hammer down which configurations can reliably produce generalizing strategies. This aligns with your feedback about moving from "the system works" to understanding what the experiments tell us.

---


