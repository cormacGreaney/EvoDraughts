# `engine_eval` — minimax tooling and a side experiment

Most of this folder supports the **reported project**: a depth-limited minimax player and `minimax_eval.py` are used only for **post-hoc comparison** with strategies evolved in the main pipeline (random and co-evolved opponents). That comparison is described in the report.

## Side experiment (not part of the reported work)

These six paths are **not** part of the project experiments or results chapters:

- `minimax_evolve_8x8.py`
- `minimax_config.py`
- `minimax_results/evolution_minimax_8x8_20260303_173549.txt`
- `minimax_results/statistics_minimax_8x8_20260303_173549.csv`
- `minimax_results/evolution_minimax_8x8_20260304_020220.txt`
- `minimax_results/statistics_minimax_8x8_20260304_020220.csv`

They correspond to a **separate exploratory run**: evolving strategies with **fitness measured against the minimax engine** (instead of the main training setup). That line of work was **left out of the reported project on purpose**, mainly because:

1. **Scope and narrative** — The thesis is centred on Grammatical Evolution with the **documented training protocol** (baseline and co-evolved opponents, standardised re-evaluation, human play, and a **short** minimax *comparison*). Training *against* minimax is a different experimental question and would need its own design and discussion which to do properly would take too much time.
2. **Fair comparison** — A shallow minimax agent is a useful **fixed reference** after evolution; using it **inside** fitness would mix two stories (search-based opponent vs evolved heuristics) and complicate how results relate to the main runs.
3. **Clarity** — Keeping minimax out of the evolutionary fitness function avoids confusing “how strong are these strategies in the main setup?” with “how well do they co-evolve against this particular search player?”

The shared minimax engine code under `minimax_engine/` and `minimax_eval.py` remains part of the **comparison** tooling for the reported work; only the **minimax-fitness evolution** script, its config, and those two saved runs are excluded from the thesis as described above.