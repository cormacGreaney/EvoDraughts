"""
Flask web application for playing draughts against evolved strategies.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from draughts_game import DraughtsBoard
from webapp.config import SECRET_KEY, DEBUG, GAME_SETTINGS
from webapp.strategy_loader import strategy_manager

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['DEBUG'] = DEBUG
CORS(app)  # Enable CORS for API endpoints


@app.route('/')
def home():
    """Homepage."""
    return render_template('home.html')


@app.route('/play')
def play():
    """Play game page."""
    return render_template('play.html')


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.route('/api/game/start', methods=['POST'])
def start_game():
    """
    Start a new game.
    Expected JSON: {'complexity': '6x6' or '8x8', 'difficulty': 'easy', 'medium', or 'hard'}
    """
    data = request.get_json()
    complexity = data.get('complexity', '6x6')
    difficulty = data.get('difficulty', 'medium')
    
    # Validate inputs
    if complexity not in ['6x6', '8x8']:
        return jsonify({'error': 'Invalid complexity. Must be 6x6 or 8x8'}), 400
    
    if difficulty not in ['easy', 'medium', 'hard']:
        return jsonify({'error': 'Invalid difficulty. Must be easy, medium, or hard'}), 400
    
    # Get strategy
    strategy_info = strategy_manager.get_best_strategy(complexity, difficulty)
    if not strategy_info:
        return jsonify({
            'error': f'No strategy found for {complexity} board with {difficulty} difficulty.'
        }), 404
    
    # Load strategy function
    ai_strategy = strategy_manager.get_strategy_function(strategy_info)
    
    # Create new board
    board_size = GAME_SETTINGS[complexity]['board_size']
    board = DraughtsBoard(board_size=board_size)
    
    # Store game state in session
    session['board'] = {
        'board_size': board_size,
        'board_state': board.board.tolist(),
        'current_player': board.current_player,
        'complexity': complexity,
        'difficulty': difficulty,
        'move_count': 0,
        'max_moves': GAME_SETTINGS[complexity]['max_moves'],
        'game_over': False,
        'winner': None
    }
    
    # Store strategy info (but not the function - can't pickle it)
    session['strategy_info'] = {
        'file': strategy_info['file'],
        'fitness': strategy_info['fitness'],
        'phenotype': strategy_info['phenotype']
    }
    
    # Get initial valid moves for human (player 1)
    valid_moves = board.get_valid_moves(1)
    
    return jsonify({
        'success': True,
        'board': board.board.tolist(),
        'current_player': board.current_player,
        'valid_moves': [[list(from_pos), list(to_pos)] for from_pos, to_pos in valid_moves],
        'game_over': board.game_over,
        'winner': board.winner,
        'strategy_info': {
            'fitness': strategy_info['fitness'],
            'description': strategy_info.get('metadata', {}).get('description', '')
        }
    })


@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    """Get current game state."""
    if 'board' not in session:
        return jsonify({'error': 'No active game'}), 404
    
    board_data = session['board']
    board = _reconstruct_board(board_data)
    
    valid_moves = board.get_valid_moves(board.current_player)
    
    return jsonify({
        'board': board.board.tolist(),
        'current_player': board.current_player,
        'valid_moves': [[list(from_pos), list(to_pos)] for from_pos, to_pos in valid_moves],
        'game_over': board.game_over,
        'winner': board.winner,
        'move_count': board_data['move_count']
    })


@app.route('/api/game/move', methods=['POST'])
def make_move():
    """
    Make a human player move.
    Expected JSON: {'from': [row, col], 'to': [row, col]}
    """
    if 'board' not in session:
        return jsonify({'error': 'No active game'}), 404
    
    data = request.get_json()
    from_pos = tuple(data.get('from'))
    to_pos = tuple(data.get('to'))
    
    # Reconstruct board from session
    board_data = session['board']
    board = _reconstruct_board(board_data)
    
    # Validate it's the human's turn (player 1)
    if board.current_player != 1:
        return jsonify({'error': 'Not your turn'}), 400
    
    # Validate move
    valid_moves = board.get_valid_moves(1)
    move_valid = any(
        (from_pos, to_pos) == (fm, tm) 
        for fm, tm in valid_moves
    )
    
    if not move_valid:
        return jsonify({'error': 'Invalid move'}), 400
    
    # Make the move
    success = board.make_move(from_pos, to_pos)
    if not success:
        return jsonify({'error': 'Move failed'}), 400
    
    board_data['move_count'] += 1
    
    # Check if game is over
    if board.game_over or board_data['move_count'] >= board_data['max_moves']:
        board.game_over = True
        if board_data['move_count'] >= board_data['max_moves']:
            # Draw by move limit - determine winner by piece count
            import numpy as np
            pieces_p1 = np.sum((board.board > 0).astype(int))
            pieces_p2 = np.sum((board.board < 0).astype(int))
            if pieces_p1 > pieces_p2:
                board.winner = 1
            elif pieces_p2 > pieces_p1:
                board.winner = -1
            else:
                board.winner = 0
        
        _save_board_to_session(board, board_data)
        
        return jsonify({
            'success': True,
            'board': board.board.tolist(),
            'current_player': board.current_player,
            'game_over': True,
            'winner': board.winner,
            'move_count': board_data['move_count']
        })
    
    # Save board state
    _save_board_to_session(board, board_data)
    
    # Now it's AI's turn (player -1)
    # Get AI move
    valid_moves_ai = board.get_valid_moves(-1)
    if not valid_moves_ai:
        board.game_over = True
        board.winner = 1  # Human wins
        _save_board_to_session(board, board_data)
        return jsonify({
            'success': True,
            'board': board.board.tolist(),
            'current_player': board.current_player,
            'game_over': True,
            'winner': 1,
            'move_count': board_data['move_count']
        })
    
    # Get strategy function
    strategy_info_data = session['strategy_info']
    # Reload the full strategy info to get the function
    complexity = board_data['complexity']
    difficulty = board_data['difficulty']
    strategy_info = strategy_manager.get_best_strategy(complexity, difficulty)
    if strategy_info and strategy_info['file'] == strategy_info_data['file']:
        strategy = strategy_manager.get_strategy_function(strategy_info)
    else:
        # Fallback: create strategy from phenotype
        from play_against_evolved import strategy_from_phenotype
        strategy = strategy_from_phenotype(
            strategy_info_data['phenotype'],
            board_data['board_size']
        )
    
    # Get features and AI move
    features = board.get_board_features()
    move_idx = strategy(features, valid_moves_ai)
    move_idx = max(0, min(move_idx, len(valid_moves_ai) - 1))
    ai_from, ai_to = valid_moves_ai[move_idx]
    
    # Make AI move
    board.make_move(ai_from, ai_to)
    board_data['move_count'] += 1
    
    # Check if game is over after AI move
    if board.game_over or board_data['move_count'] >= board_data['max_moves']:
        board.game_over = True
        if board_data['move_count'] >= board_data['max_moves']:
            import numpy as np
            pieces_p1 = np.sum((board.board > 0).astype(int))
            pieces_p2 = np.sum((board.board < 0).astype(int))
            if pieces_p1 > pieces_p2:
                board.winner = 1
            elif pieces_p2 > pieces_p1:
                board.winner = -1
            else:
                board.winner = 0
        
        _save_board_to_session(board, board_data)
    
    # Save board state
    _save_board_to_session(board, board_data)
    
    # Get valid moves for human
    valid_moves = board.get_valid_moves(1) if not board.game_over else []
    
    return jsonify({
        'success': True,
        'board': board.board.tolist(),
        'current_player': board.current_player,
        'valid_moves': [[list(fm), list(tm)] for fm, tm in valid_moves],
        'ai_move': {
            'from': list(ai_from),
            'to': list(ai_to)
        },
        'game_over': board.game_over,
        'winner': board.winner,
        'move_count': board_data['move_count']
    })


def _reconstruct_board(board_data: dict) -> DraughtsBoard:
    """Reconstruct a DraughtsBoard object from session data."""
    import numpy as np
    board = DraughtsBoard(board_size=board_data['board_size'])
    board.board = np.array(board_data['board_state'])
    board.current_player = board_data['current_player']
    board.game_over = board_data.get('game_over', False)
    board.winner = board_data.get('winner', None)
    return board


def _save_board_to_session(board: DraughtsBoard, board_data: dict):
    """Save board state to session data."""
    board_data['board_state'] = board.board.tolist()
    board_data['current_player'] = board.current_player
    board_data['game_over'] = board.game_over
    board_data['winner'] = board.winner
    session['board'] = board_data


@app.route('/api/strategies/available', methods=['GET'])
def get_available_strategies():
    """Get list of available strategies by complexity and difficulty."""
    complexity = request.args.get('complexity', '6x6')
    
    if complexity not in ['6x6', '8x8']:
        return jsonify({'error': 'Invalid complexity'}), 400
    
    result = {}
    for difficulty in ['easy', 'medium', 'hard']:
        strategies = strategy_manager.get_strategies_by_difficulty(complexity, difficulty)
        result[difficulty] = {
            'count': len(strategies),
            'strategies': [
                {
                    'fitness': s['fitness'],
                    'file': os.path.basename(s['file'])
                }
                for s in strategies[:5]  # Limit to top 5
            ]
        }
    
    return jsonify(result)


if __name__ == '__main__':
    # Reload strategies on startup
    strategy_manager.reload_strategies()
    print("Available strategies loaded:")
    for complexity in ['6x6', '8x8']:
        for difficulty in ['easy', 'medium', 'hard']:
            count = len(strategy_manager.get_strategies_by_difficulty(complexity, difficulty))
            if count > 0:
                print(f"  {complexity} {difficulty}: {count} strategies")
    
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)

