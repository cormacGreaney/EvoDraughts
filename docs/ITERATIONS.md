# Multiple Iterations

The project supports multiple iterations with increasing game complexity.

## Iteration 1: 6x6 Simplified (Current)

**Status**: Fully working, three runs completed

- Board: 6x6
- Features: 47 (36 positions + 11 game state)
- Max moves: 100

I started here because it's faster to simulate and easier to understand. Games finish quickly, so I can run 100 generations in 2-3 hours.

**Results**: Strategies evolving successfully, win rates improving (20% → 40%), fitness improving (0.45 → 0.275).

**Configuration**: `CURRENT_ITERATION = ITERATION_1` in `config.py`

## Iteration 2: 8x8 Standard (Future)

**Status**: Defined but not yet tested

- Board: 8x8
- Features: 75 (64 positions + 11 game state)
- Max moves: 200

When ready, I'll switch to this for more strategic depth. Will need to update the grammar file to include features x[0] through x[74].

**Configuration**: Change to `CURRENT_ITERATION = ITERATION_2` in `config.py`

## Why Multiple Iterations

- Start simple to verify the system works
- Increase complexity once I understand the basics
- Compare strategies across different board sizes
- 6x6 is fast for experimentation, 8x8 for deeper analysis

The system is designed to easily switch between iterations - just change the configuration.
