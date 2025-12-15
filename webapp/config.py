"""
Configuration for the web application.
Maps difficulty levels to strategy files and defines game settings.
"""

import os
from pathlib import Path

# Get the project root directory (parent of webapp folder)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Strategy difficulty mapping
# Strategies are mapped based on their fitness (lower fitness = better strategy)
# You can manually assign specific strategy files to difficulty levels here
STRATEGY_MAPPING = {
    '6x6': {
        'easy': {
            # Weaker strategies (higher fitness, lower win rate)
            'fitness_range': (0.55, 1.0),
            'description': 'Easy strategies'
        },
        'medium': {
            'fitness_range': (0.40, 0.55),
            'description': 'Medium strategies'
        },
        'hard': {
            # Stronger strategies (lower fitness, higher win rate)
            'fitness_range': (0.0, 0.40),
            'description': 'Hard strategies'
        }
    },
    '8x8': {
        'easy': {
            'fitness_range': (0.55, 1.0),
            'description': 'Easy strategies'
        },
        'medium': {
            'fitness_range': (0.40, 0.55),
            'description': 'Medium strategies'
        },
        'hard': {
            'fitness_range': (0.0, 0.40),
            'description': 'Hard strategies'
        }
    }
}

# Game settings
GAME_SETTINGS = {
    '6x6': {
        'board_size': 6,
        'max_moves': 100
    },
    '8x8': {
        'board_size': 8,
        'max_moves': 200
    }
}

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

