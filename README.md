# EvoDraughts: Evolving Draughts Strategies Using GRAPE

This project uses Grammatical Evolution (GE) via the GRAPE framework to evolve draughts playing strategies. Strategies are represented as expressions in a BNF grammar that evaluate board positions and select moves.

## Current Status

System is fully implemented and tested. I've completed three evolution runs showing clear improvement:
- **Run 1**: 20% win rate, simple strategies
- **Run 2**: 30% win rate, still simple
- **Run 3**: 40% win rate, complex conditional strategies

**Latest Results:**
- Best Strategy: `if_(greater_than_or_equal(x[0], x[25]), x[19], x[20])`
- Fitness: 0.325 (improved from 0.45)
- Win Rate: 40% vs random
- Evolution: Clear improvement over 100 generations

## Quick Start

```bash
pip install -r requirements.txt
python evolve_draughts.py
```

Results are saved to `results/` folder automatically.

## How It Works

1. **Grammar**: BNF grammar defines possible strategies as expressions
2. **Board Features**: 47 features for 6x6 board (positions + game state)
3. **Evaluation**: Strategies play games against random opponent
4. **Evolution**: GRAPE evolves better strategies over generations
5. **Results**: Automatically saved with best strategy and statistics

## Project Structure

- `evolve_draughts.py` - Main evolution script
- `draughts_game.py` - Game implementation
- `grammars/draughts.bnf` - Strategy grammar
- `config.py` - Settings
- `results/` - All run results

## Documentation

All documentation is in the `docs/` folder:
- `docs/HOW_TO_RUN.md` - Running the evolution
- `docs/SETTINGS.md` - Settings and what I learned
- `docs/SETUP_COMPLETE.md` - Project status
- `docs/ITERATIONS.md` - Multiple iteration support

## Key Features

- Complete draughts game implementation
- GRAPE integration for grammatical evolution
- Grammar-defined strategy space
- Fitness evaluation through game simulation
- Automatic result saving
- Strategies using grammar constructs correctly
- Clear evidence of evolution over generations

The system works and produces evolving strategies using GE with the defined grammar.
