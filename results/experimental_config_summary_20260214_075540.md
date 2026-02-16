# Experimental Configuration Summary

Generated: 2026-02-14 07:55:40

This document provides a comprehensive summary of all experimental runs conducted for the EvoDraughts project.

## Summary Statistics

- **Total Runs**: 10
- **6x6 Runs**: 8
- **8x8 Runs**: 2
- **Unique Configurations**: 6

## Configuration Summary

| Config ID | Board | Pop | Gens | Games/Eval | Co-Evol | Co-Evol Opps | # Runs | Description |
|-----------|-------|-----|------|-----------|---------|--------------|--------|-------------|
| Config_1 | 6x6 | 100 | 10 | 10 | No | - | 1 | 6x6 baseline (random opponents) |
| Config_2 | 6x6 | 150 | 30 | 15 | Yes | 5 | 1 | 6x6 with co-evolution |
| Config_3 | 6x6 | 200 | 100 | 20 | No | - | 2 | 6x6 baseline (random opponents) |
| Config_4 | 6x6 | 300 | 50 | 25 | Yes | 5 | 4 | 6x6 with co-evolution |
| Config_5 | 8x8 | 300 | 50 | 20 | Yes | 5 | 1 | 8x8 with co-evolution |
| Config_6 | 8x8 | 400 | 75 | 20 | Yes | 5 | 1 | 8x8 with co-evolution |

## Test Evaluation Protocol

| Board Size | Random Seed | Test Games | Opponent Type | Notes |
|------------|-------------|------------|---------------|-------|
| 6x6 | 999 | 20 | Random | Standardized for all 6x6 strategies |
| 8x8 | 888 | 20 | Random | Standardized for all 8x8 strategies |

**Note**: All strategies of the same board size are tested against the same random opponent sequence (using fixed random seed) to ensure fair comparison.

## Detailed Run Information

| Run ID | Config | Result File | Date | Best Strategy | Training Fitness | Test Win Rate |
|--------|-------|-------------|------|---------------|------------------|---------------|
| 20251213_034131 | Config_1 | evolution_20251213_034131.txt | 2025-12-13 03:41:31 | `x[32]` | 0.3000 | 20.0% |
| 20251213_055102 | Config_3 | evolution_20251213_055102.txt | 2025-12-13 05:51:02 | `x[8]` | 0.2750 | 30.0% |
| 20251213_083718 | Config_3 | evolution_20251213_083718.txt | 2025-12-13 08:37:18 | `if_(greater_than_or_equal(x[0], x[25]), x[19], ...` | 0.3250 | 30.0% |
| 20251214_220441 | Config_2 | evolution_20251214_220441.txt | 2025-12-14 22:04:41 | `x[32]` | 0.2000 | 30.0% |
| 20251215_032724 | Config_4 | evolution_20251215_032724.txt | 2025-12-15 03:27:24 | `min_(x[45], 2.0)` | N/A | 65.0% |
| 20251215_090842 | Config_4 | evolution_20251215_090842.txt | 2025-12-15 09:08:42 | `max_(if_(less_than_or_equal(x[6], 0.1), 2.0, x[...` | N/A | 0.0% |
| 20251215_202804 | Config_4 | evolution_20251215_202804.txt | 2025-12-15 20:28:04 | `x[20]` | 0.2300 | 25.0% |
| 20251216_034713 | Config_4 | evolution_20251216_034713.txt | 2025-12-16 03:47:13 | `add(x[20], mul(x[45], x[40]))` | 0.1800 | 75.0% |
| 8x8_20251216_222007 | Config_5 | evolution_8x8_20251216_222007.txt | 2025-12-16 22:20:07 | `mul(x[33], x[21])` | 0.1500 | 10.0% |
| 8x8_20251218_002100 | Config_6 | evolution_8x8_20251218_002100.txt | 2025-12-18 00:21:00 | `add(x[35], x[26])` | 0.1750 | 80.0% |

## Configuration Statistics

### Config_1: 6x6 Board

- **Population Size**: 100
- **Generations**: 10
- **Games per Evaluation**: 10
- **Co-evolution**: No
- **Number of Runs**: 1
- **Test Win Rate**: Mean=20.0%, Std=0.0%, Range=[20.0%, 20.0%]
- **Runs**: evolution_20251213_034131.txt

### Config_2: 6x6 Board

- **Population Size**: 150
- **Generations**: 30
- **Games per Evaluation**: 15
- **Co-evolution**: Yes
- **Co-evolution Opponents**: 5
- **Number of Runs**: 1
- **Test Win Rate**: Mean=30.0%, Std=0.0%, Range=[30.0%, 30.0%]
- **Runs**: evolution_20251214_220441.txt

### Config_3: 6x6 Board

- **Population Size**: 200
- **Generations**: 100
- **Games per Evaluation**: 20
- **Co-evolution**: No
- **Number of Runs**: 2
- **Test Win Rate**: Mean=30.0%, Std=0.0%, Range=[30.0%, 30.0%]
- **Runs**: evolution_20251213_055102.txt, evolution_20251213_083718.txt

### Config_4: 6x6 Board

- **Population Size**: 300
- **Generations**: 50
- **Games per Evaluation**: 25
- **Co-evolution**: Yes
- **Co-evolution Opponents**: 5
- **Number of Runs**: 4
- **Test Win Rate**: Mean=41.2%, Std=30.3%, Range=[0.0%, 75.0%]
- **Runs**: evolution_20251215_032724.txt, evolution_20251215_090842.txt, evolution_20251215_202804.txt, evolution_20251216_034713.txt

### Config_5: 8x8 Board

- **Population Size**: 300
- **Generations**: 50
- **Games per Evaluation**: 20
- **Co-evolution**: Yes
- **Co-evolution Opponents**: 5
- **Number of Runs**: 1
- **Test Win Rate**: Mean=10.0%, Std=0.0%, Range=[10.0%, 10.0%]
- **Runs**: evolution_8x8_20251216_222007.txt

### Config_6: 8x8 Board

- **Population Size**: 400
- **Generations**: 75
- **Games per Evaluation**: 20
- **Co-evolution**: Yes
- **Co-evolution Opponents**: 5
- **Number of Runs**: 1
- **Test Win Rate**: Mean=80.0%, Std=0.0%, Range=[80.0%, 80.0%]
- **Runs**: evolution_8x8_20251218_002100.txt

## Notes

- All test evaluations use standardized random seeds for fair comparison
- Training fitness is calculated during evolution (may use different opponents for co-evolution runs)
- Test win rate is against standardized random opponent
- Some runs may have missing data if result files are incomplete

