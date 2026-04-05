# Experimental Configuration Summary

Generated: 2026-04-05 21:40:29

This document provides a comprehensive summary of all experimental runs conducted for the EvoDraughts project.

## Summary Statistics

- **Total Runs**: 25
- **6x6 Runs**: 8
- **8x8 Runs**: 17
- **Unique Configurations**: 6

## Configuration Summary

| Config ID | Board | Pop | Gens | Games/Eval | Co-Evol | Co-Evol Opps | # Runs | Description |
|-----------|-------|-----|------|-----------|---------|--------------|--------|-------------|
| Config_1 | 6x6 | 100 | 10 | 10 | No | - | 1 | 6x6 baseline (random opponents) |
| Config_2 | 6x6 | 150 | 30 | 15 | Yes | 5 | 1 | 6x6 with co-evolution |
| Config_3 | 6x6 | 200 | 100 | 20 | No | - | 2 | 6x6 baseline (random opponents) |
| Config_4 | 6x6 | 300 | 50 | 25 | Yes | 5 | 4 | 6x6 with co-evolution |
| Config_5 | 8x8 | 300 | 50 | 20 | Yes | 5 | 1 | 8x8 with co-evolution |
| Config_6 | 8x8 | 400 | 75 | 20 | Yes | 5 | 16 | 8x8 with co-evolution |

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
| 8x8_20260216_052949 | Config_6 | evolution_8x8_20260216_052949.txt | 2026-02-16 05:29:49 | `sub(add(if_(less_than_or_equal(5.0, 2.0), -0.1,...` | 0.1250 | 75.0% |
| 8x8_20260218_003143 | Config_6 | evolution_8x8_20260218_003143.txt | 2026-02-18 00:31:43 | `add(if_(greater_than_or_equal(mul(x[49], 10.0),...` | 0.0750 | 85.0% |
| 8x8_20260219_203646 | Config_6 | evolution_8x8_20260219_203646.txt | 2026-02-19 20:36:46 | `mul(x[73], x[24])` | 0.0500 | 90.0% |
| 8x8_20260223_045614 | Config_6 | evolution_8x8_20260223_045614.txt | 2026-02-23 04:56:14 | `mul(add(abs_(10.0), x[26]), x[23])` | 0.0750 | 76.0% |
| 8x8_20260225_114105 | Config_6 | evolution_8x8_20260225_114105.txt | 2026-02-25 11:41:05 | `1.0` | 0.1000 | 69.3% |
| 8x8_20260227_011809 | Config_6 | evolution_8x8_20260227_011809.txt | 2026-02-27 01:18:09 | `add(if_(greater_than_or_equal(x[35], x[64]), x[...` | 0.1000 | 66.7% |
| 8x8_20260302_115155 | Config_6 | evolution_8x8_20260302_115155.txt | 2026-03-02 11:51:55 | `add(x[24], x[68])` | 0.0750 | 72.0% |
| 8x8_20260307_093010 | Config_6 | evolution_8x8_20260307_093010.txt | 2026-03-07 09:30:10 | `sub(x[53], neg(x[70]))` | 0.0500 | 73.3% |
| 8x8_20260309_155326 | Config_6 | evolution_8x8_20260309_155326.txt | 2026-03-09 15:53:26 | `add(x[68], sub(x[24], x[46]))` | 0.0500 | 81.3% |
| 8x8_20260311_043951 | Config_6 | evolution_8x8_20260311_043951.txt | 2026-03-11 04:39:51 | `add(add(x[40], x[39]), x[19])` | 0.0500 | 78.0% |
| 8x8_20260312_214315 | Config_6 | evolution_8x8_20260312_214315.txt | 2026-03-12 21:43:15 | `sub(x[70], x[46])` | 0.0750 | 70.0% |
| 8x8_20260316_010459 | Config_6 | evolution_8x8_20260316_010459.txt | 2026-03-16 01:04:59 | `mul(x[72], x[39])` | 0.0250 | 63.3% |
| 8x8_20260318_051106 | Config_6 | evolution_8x8_20260318_051106.txt | 2026-03-18 05:11:06 | `add(x[19], min_(x[54], if_(greater_than_or_equa...` | 0.1000 | 78.0% |
| 8x8_20260320_104123 | Config_6 | evolution_8x8_20260320_104123.txt | 2026-03-20 10:41:23 | `sub(x[19], sub(x[35], x[24]))` | 0.0750 | 81.3% |
| 8x8_20260322_094426 | Config_6 | evolution_8x8_20260322_094426.txt | 2026-03-22 09:44:26 | `sub(if_(greater_than_or_equal(x[21], max_(-1.0,...` | 0.1000 | 68.7% |

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
- **Number of Runs**: 16
- **Test Win Rate**: Mean=75.5%, Std=6.9%, Range=[63.3%, 90.0%]
- **Runs**: evolution_8x8_20251218_002100.txt, evolution_8x8_20260216_052949.txt, evolution_8x8_20260218_003143.txt, evolution_8x8_20260219_203646.txt, evolution_8x8_20260223_045614.txt, evolution_8x8_20260225_114105.txt, evolution_8x8_20260227_011809.txt, evolution_8x8_20260302_115155.txt, evolution_8x8_20260307_093010.txt, evolution_8x8_20260309_155326.txt, evolution_8x8_20260311_043951.txt, evolution_8x8_20260312_214315.txt, evolution_8x8_20260316_010459.txt, evolution_8x8_20260318_051106.txt, evolution_8x8_20260320_104123.txt, evolution_8x8_20260322_094426.txt

## Notes

- All test evaluations use standardized random seeds for fair comparison
- Training fitness is calculated during evolution (may use different opponents for co-evolution runs)
- Test win rate is against standardized random opponent
- Some runs may have missing data if result files are incomplete

