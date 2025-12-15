"""
Simple startup script for the web application.
Run this to start the Flask server.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import app
from webapp.strategy_loader import strategy_manager

if __name__ == '__main__':
    # Reload strategies on startup
    print("Loading strategies...")
    strategy_manager.reload_strategies()
    
    print("\nAvailable strategies:")
    for complexity in ['6x6', '8x8']:
        for difficulty in ['easy', 'medium', 'hard']:
            count = len(strategy_manager.get_strategies_by_difficulty(complexity, difficulty))
            if count > 0:
                best = strategy_manager.get_best_strategy(complexity, difficulty)
                fitness = best['fitness'] if best else 'N/A'
                print(f"  {complexity} {difficulty}: {count} strategies (best fitness: {fitness})")
    
    print("\n" + "="*60)
    print("Starting EvoDraughts Web Application")
    print("="*60)
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

