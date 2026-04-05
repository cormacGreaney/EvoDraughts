# EvoDraughts

## Evolving game strategies using Grammatical Evolution

**Cormac Greaney / 22352228**

Department of Computer Science and Information Systems
Faculty of Science and Engineering
University of Limerick

Submitted to the University of Limerick for the degree of
BSc. in Computer Systems academic year 2025/26

Supervisor: Dr. Conor Ryan

## Abstract

Designing effective strategies for complex board games traditionally relies on domain expertise and manually engineered evaluation functions. This project, EvoDraughts, investigates an alternative approach: evolving draughts strategies using Grammatical Evolution (GE). Candidate strategies are represented as grammar-constrained symbolic expressions and evaluated through repeated gameplay rather than manual design.

The system integrates a custom draughts engine supporting both 6×6 and 8×8 boards, a grammar-based representation ensuring syntactic validity and interpretability, evolutionary search implemented using the GRAPE framework, and a web application for human–agent gameplay. Finally, a separate minimax-based agent was used for a short classical comparison where a set of representative 8×8 evolved strategies were played against it.

Across multiple experimental runs, evolved strategies consistently outperform a random baseline, with standardised test win rates on the 8×8 board reaching up to 94.7% under a fixed multi-seed evaluation protocol. Several strategies achieve win rates above 80%, demonstrating that the evolutionary framework is capable of producing effective decision-making behaviour within the defined search space. However, performance varies across configurations, and some strategies exhibit evidence of overfitting and limited generalisation.

Human evaluation shows that while evolved strategies are capable of winning games, they remain consistently beatable by human players and are generally perceived as moderately challenging rather than difficult. This highlights a gap between strong performance against simple baseline opponents and the ability to produce robust, high-level gameplay.

Overall, the results demonstrate that Grammatical Evolution can generate functional and interpretable draughts strategies, while also highlighting the limitations of grammar-based evolutionary approaches in adversarial domains. The project provides a complete experimental framework for evolving, evaluating, and analysing game-playing strategies.


See the accompanying report for further information

## Key results (summary)

- **Standardised testing (8×8):** Evolved strategies consistently beat a random baseline; multi-seed evaluation reached up to **94.7%** win rate, with several strategies above **80%**; mean test win rate across main 8×8 runs was **73.6%**. Performance varies by configuration; some strategies show train–test gaps consistent with overfitting.
- **Human evaluation:** Tested agents remained largely **beatable** by human players and were rated **moderately challenging** on average-strong scores against simple baselines do not imply strong play against people.
- **Minimax comparison:** Brief post-hoc runs at fixed depth illustrate **opponent-dependent** outcomes; interpret alongside deterministic-play notes in the report.

## Requirements

- Python 3.x  
- Dependencies: root `requirements.txt`; the web app has its own `webapp/requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

For the web interface:

```bash
cd webapp
pip install -r requirements.txt
```

## Usage

**Run evolution** (from the repository root):

```bash
python evolve_draughts.py        # 6×6 (initial experiments)
python evolve_draughts2.py       # 6×6 (second phase experiments)
python evolve_draughts_8x8.py    # 8×8 (main experiments)
```

Settings are centralised in `config.py`.

**Standardised re-evaluation** of saved runs (see report for the fixed seed/game protocol):

```bash
python reevaluate_strategies_match_evolution.py
```

**Play against saved strategies** (CLI):

```bash
python play_against_evolved.py --list
```

**Web interface:**

```bash
cd webapp
python run.py
```

Open the URL shown in the terminal (`http://localhost:5000`).

**Minimax comparison** (optional): see `engine_eval/minimax_eval.py` and the report for setup and interpretation.

## Project layout

- `evolve_draughts.py` / `evolve_draughts2.py` / `evolve_draughts_8x8.py` — evolution entry points (6×6 vs 8×8; see report for semantics)  
- `draughts_game.py` — game rules and engine  
- `draughts_functions.py` — safe operators for grammar expressions  
- `config.py` — configuration and iteration presets  
- `grammars/` — BNF grammars for strategy expressions  
- `reevaluate_strategies_match_evolution.py` — standardised post-training evaluation aligned with evolution  
- `results/` — evolution outputs, standardised tests, and analysis artefacts  
- `webapp/` — Flask app for human vs evolved play  
- `engine_eval/` — minimax-style opponent and evaluation module  
- `docs/` — supplementary documentation (e.g. 8×8 plan, run tracker)  
- `grape/` — GRAPE library used for grammatical evolution  
