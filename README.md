# EvoDraughts

Note: the contents of docs/ folder are not up to date right now

Using Grammatical Evolution (via GRAPE) to evolve draughts playing strategies. The strategies are basically math expressions that evaluate board positions and pick moves - and they get better over time through evolution.

## What I've Got

- **6x6 version**: Works pretty well, got strategies winning 75%+ against random
- **8x8 version**: Still working on this one - had some overfitting issues but making progress
- **Web app**: Simple Flask interface to play against the evolved strategies (in `webapp/`)
- **Multiple runs**: Been tweaking parameters and running evolutions, results are in `results/`

## Quick Start

```bash
pip install -r requirements.txt
python evolve_draughts.py          # 6x6 board
python evolve_draughts_8x8.py       # 8x8 board
```

Or play against evolved strategies:
```bash
python play_against_evolved.py --list
```

Or use the web interface:
```bash
cd webapp
pip install -r requirements.txt
python run.py
```

## How It Works

The system uses GRAPE (Grammatical Evolution framework) to evolve strategies. Each strategy is a grammar expression that takes board features and outputs a move. They play games, get evaluated, and the best ones survive to the next generation.

## Files

- `evolve_draughts.py` - Original 6x6 evolution script
- `evolve_draughts_8x8.py` - 8x8 version with co-evolution
- `draughts_game.py` - Game logic
- `config.py` - All the settings and parameters
- `grammars/` - BNF grammars defining the strategy space
- `results/` - some of my evolution runs and analysis
- `webapp/` - Web interface for playing

