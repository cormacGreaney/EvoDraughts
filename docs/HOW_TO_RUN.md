# How to Run

## Quick Start

1. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

2. **Test setup (optional):**
```powershell
python test_setup.py
```

3. **Run evolution:**
```powershell
python evolve_draughts.py
```

Or save output to file:
```powershell
python evolve_draughts.py > my_run.txt 2>&1
```

## What Happens

The script creates 200 random strategies, evaluates each by playing 20 games, selects the best, creates a new generation, and repeats for 100 generations. Takes about 2-3 hours.

## Results

Results are automatically saved to `results/` folder:
- `evolution_TIMESTAMP.txt` - Best strategy, fitness, test results
- `statistics_TIMESTAMP.csv` - Generation-by-generation data

## Testing Evolved Strategies

```powershell
# List all strategies
python play_against_evolved.py --list

# Test a specific strategy
python play_against_evolved.py --file results/evolution_TIMESTAMP.txt --games 20
```

## Monitoring

**Check if running:**
```powershell
Get-Process python -ErrorAction SilentlyContinue
```

**Watch log file:**
```powershell
Get-Content my_run.txt -Wait -Tail 50
```

## Understanding Output

**Fitness**: Lower is better (0.0 = perfect). My runs: 0.45 → 0.275

**Strategy Examples:**
- Simple: `x[8]` (just a feature lookup)
- Complex: `if_(greater_than_or_equal(x[0], x[25]), x[19], x[20])` (conditional logic)

**Win Rate**: Latest run achieved 40% vs random. Note: With only 10 test games, there's variance between runs.


