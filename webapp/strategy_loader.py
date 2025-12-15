"""
Utility module for loading and managing evolved strategies.
Handles strategy discovery, loading, and difficulty categorization.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import game modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from play_against_evolved import load_strategy_from_file, strategy_from_phenotype
from webapp.config import RESULTS_DIR, STRATEGY_MAPPING, GAME_SETTINGS


class StrategyManager:
    """Manages loading and selection of evolved strategies."""
    
    def __init__(self):
        self._strategies_cache: Dict[str, Dict] = {}
        self._load_all_strategies()
    
    def _load_all_strategies(self):
        """Scan results directory and cache all available strategies."""
        if not RESULTS_DIR.exists():
            print(f"Warning: Results directory not found: {RESULTS_DIR}")
            return
        
        # Find all evolution result files
        result_files = list(RESULTS_DIR.glob("evolution_*.txt"))
        
        for result_file in result_files:
            try:
                phenotype, board_size, fitness, metadata = load_strategy_from_file(str(result_file))
                
                # Determine board complexity
                complexity = f"{board_size}x{board_size}"
                
                # Cache strategy info (don't load the actual function yet - lazy loading)
                strategy_key = f"{complexity}_{result_file.stem}"
                
                self._strategies_cache[strategy_key] = {
                    'file': str(result_file),
                    'complexity': complexity,
                    'board_size': board_size,
                    'fitness': fitness,
                    'phenotype': phenotype,
                    'metadata': metadata
                }
            except Exception as e:
                print(f"Warning: Could not load strategy from {result_file}: {e}")
    
    def get_strategies_by_difficulty(self, complexity: str, difficulty: str) -> List[Dict]:
        """
        Get all strategies matching the specified complexity and difficulty.
        
        Args:
            complexity: '6x6' or '8x8'
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            List of strategy dictionaries
        """
        if difficulty not in STRATEGY_MAPPING.get(complexity, {}):
            return []
        
        fitness_range = STRATEGY_MAPPING[complexity][difficulty]['fitness_range']
        min_fitness, max_fitness = fitness_range
        
        matching_strategies = []
        for key, strategy_info in self._strategies_cache.items():
            if strategy_info['complexity'] != complexity:
                continue
            
            fitness = strategy_info.get('fitness')
            if fitness is None:
                continue
            
            if min_fitness <= fitness <= max_fitness:
                matching_strategies.append(strategy_info)
        
        # Sort by fitness (lower is better, so best strategies first)
        matching_strategies.sort(key=lambda x: x.get('fitness', 1.0))
        
        return matching_strategies
    
    def get_strategy_function(self, strategy_info: Dict):
        """
        Get a callable strategy function from strategy info.
        Uses lazy loading with caching.
        
        Args:
            strategy_info: Strategy dictionary from get_strategies_by_difficulty
            
        Returns:
            Callable strategy function
        """
        # Create a cache key
        cache_key = f"{strategy_info['complexity']}_{strategy_info['file']}"
        
        # Check if we've already loaded this strategy
        if 'function' not in strategy_info:
            try:
                strategy_info['function'] = strategy_from_phenotype(
                    strategy_info['phenotype'],
                    strategy_info['board_size']
                )
            except Exception as e:
                print(f"Error loading strategy function: {e}")
                # Return a random strategy as fallback
                import random
                def fallback_strategy(features, valid_moves):
                    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0
                strategy_info['function'] = fallback_strategy
        
        return strategy_info['function']
    
    def get_best_strategy(self, complexity: str, difficulty: str) -> Optional[Dict]:
        """
        Get the best strategy for a given complexity and difficulty.
        Returns the strategy with the lowest fitness (best performance).
        
        Args:
            complexity: '6x6' or '8x8'
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            Strategy dictionary or None if no strategy found
        """
        strategies = self.get_strategies_by_difficulty(complexity, difficulty)
        if strategies:
            # Already sorted by fitness, return the first (best) one
            return strategies[0]
        return None
    
    def reload_strategies(self):
        """Reload all strategies from the results directory."""
        self._strategies_cache.clear()
        self._load_all_strategies()


# Global strategy manager instance
strategy_manager = StrategyManager()

