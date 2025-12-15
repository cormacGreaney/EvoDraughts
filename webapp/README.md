# EvoDraughts Web App

Simple web interface for playing draughts against my evolved strategies. Built with Flask.

## Quick Start

```bash
cd webapp
pip install -r requirements.txt
python run.py
```

Then open http://localhost:5000 in your browser.

## What It Does

- Lets you pick board size (6x6 or 8x8) and difficulty (easy/medium/hard)
- Loads strategies from the `results/` folder
- You play as red, the evolved strategy plays as black
- Click your pieces and then click where to move them

That's basically it. Games aren't being recorded right now - it's just for testing how the strategies perform against humans next semester
