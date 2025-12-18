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
# Manually assign specific result files to each difficulty level
STRATEGY_MAPPING = {
    '6x6': {
        'easy': [
            'evolution_20251214_220441.txt',
        ],
        'medium': [
            'evolution_20251215_032724.txt',
        ],
        'hard': [
            'evolution_20251216_034713.txt',
        ]
    },
    '8x8': {
        'easy': [
            'evolution_8x8_20251216_222007.txt',
        ],
        'medium': [
            'evolution_8x8_20251218_002100.txt',
        ],
        'hard': [
        ]
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

