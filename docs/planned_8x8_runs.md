# Planned 8x8 Experimental Runs

## Feedback Summary

- Avoid single-shot runs: need **multiple replicates per condition** (2–3 minimum; 3+ better) to claim X is better than Y. Report must state this.
- **Narrow conditions**: choose only **two** co-evolution ratio settings (with replicates each) and **two** mutation settings (with replicates each).
- **Test protocol**: multiple test seeds (5–10), more test games (30+). If time is tight, reduce number of opponents but increase games per opponent and **keep the opponent set fixed**.
- **Tracking**: one spreadsheet row per run with Run ID, config, evolution seed(s), training protocol, test protocol (opponent suite ID, seeds, games), primary metrics (win-rate with CI, train–test gap, variance across seeds), secondary metrics (complexity, strategy size).

---

## Completed: Runs 1–5

- **Runs 1–2**: Initial 8x8 (Config_5, Config_6).
- **Runs 3–5**: Three replicates of Config_6 (40% co-evolution, mutation 0.15), each with a **different evolution seed** (456, 789, 1011) so they are independent. Kept as planned.

---

## Revised plan: Runs 6 onward

### Evolution seeds

- **Each run = one evolution seed.** Replicates of the same condition use **different** evolution seeds (e.g. 2001, 2002, 2003 for three replicates of “30% co-evolution”). Same seed would just repeat the same run; different seeds give independent evolutionary paths for fair comparison.

---

### 1. Co-evolution ratio (two settings, 3 replicates each)

**Condition A – Lower co-evolution (more random)**  

- Co-evolution ratio: **30%** co-evolved, 70% random  
- All else same as Config_6 (pop 400, gen 75, games/eval 20, mutation 0.15, complexity 0.20).  
- **Replicates**: 3 runs, each with a **different evolution seed** (e.g. 2001, 2002, 2003).

**Condition B – Higher co-evolution**  

- Co-evolution ratio: **50%** co-evolved, 50% random  
- Same base config as above.  
- **Replicates**: 3 runs, each with a **different evolution seed** (e.g. 2011, 2012, 2013).

We compare A vs B (and vs Config_6 baseline from runs 3–5) using **replicate-level** metrics (e.g. mean win rate, CI, variance). Report will state that we use 3 replicates per condition (3+ would be better; constraint is run length).

---

### 2. Mutation rate (two settings, 3 replicates each)

**Condition C – Lower mutation**  

- Mutation rate: **0.10** (Config_6 uses 0.15).  
- All else same as Config_6.  
- **Replicates**: 3 runs, different evolution seeds (e.g. 3011, 3012, 3013).

**Condition D – Higher mutation**  

- Mutation rate: **0.20**.  
- Same base config.  
- **Replicates**: 3 runs, different evolution seeds (e.g. 3021, 3022, 3023).

Same idea: compare C vs D (and vs Config_6) with replicate-level stats; report will own the 3 replicate count.

---

---

## Summary table (revised runs)


| Run group | Condition         | Replicates | Evolution seeds (example) |
| --------- | ----------------- | ---------- | ------------------------- |
| 3–5       | Config_6 baseline | 3          | 456, 789, 1011 (done)     |
| 6–8       | Co-evol 30% (A)   | 3          | 2001, 2002, 2003          |
| 9-11      | Co-evol 50% (B)   | 3          | 2011, 2012, 2013          |
| 12-14     | Mutation 0.10 (C) | 3          | 3011, 3012, 3013          |
| 15-17     | Mutation 0.20 (D) | 3          | 3021, 3022, 3023          |


---

## Notes

- Same config + different evolution seed = independent replicate. We do not reuse evolution seeds across runs.  
- All runs tracked in the experiment spreadsheet (Run ID, config, evolution seed(s), training protocol, test protocol, primary and secondary metrics).

