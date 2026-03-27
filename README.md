# EvoDraughts

EvoDraughts uses **grammatical evolution** (via the GRAPE library in `grape/`) to evolve draughts-playing strategies. Strategies are expressions generated from a grammar; they are evaluated by playing games, and better performers are selected and varied over successive generations.

The repository includes **6×6** and **8×8** board setups, evolution scripts, a **Flask web app** to play against evolved strategies, and **results** from experiments. Additional notes (including an 8×8 experimental plan and run tracker) are in `docs/`.

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
python evolve_draughts.py      # 6×6
python evolve_draughts_8x8.py  # 8×8
```

Settings are centralised in `config.py`.

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

## Project layout

- `evolve_draughts.py` / `evolve_draughts_8x8.py` — evolution entry points  
- `draughts_game.py` — game rules and engine  
- `config.py` — configuration and iteration presets  
- `grammars/` — BNF grammars for strategy expressions  
- `results/` — evolution outputs and evaluation summaries  
- `webapp/` — Flask app for human vs evolved play  
- `docs/` — supplementary documentation (e.g. 8×8 plan, run tracker)  
- `grape/` — GRAPE library used for grammatical evolution

