"""
Draughts (Checkers) Game Implementation
A simple implementation of English draughts for use with GRAPE evolutionary algorithm.
"""

import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy


class DraughtsBoard:
    """Represents a draughts board state."""
    
    # Board representation:
    # 0 = empty
    # 1 = player 1 (red) piece
    # 2 = player 1 (red) king
    # -1 = player 2 (black) piece
    # -2 = player 2 (black) king
    
    def __init__(self, board_size=6):
        """
        Initialize a draughts board.
        
        Args:
            board_size: Size of the board (default 6 for simplified version, can be 8 for standard)
        """
        self.BOARD_SIZE = board_size
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=int)
        self._setup_initial_board()
        self.current_player = 1  # 1 for player 1 (red), -1 for player 2 (black)
        self.game_over = False
        self.winner = None
        
    def _setup_initial_board(self):
        """Set up the initial board configuration."""
        # Calculate number of rows for each player based on board size
        # For 6x6: 2 rows each, for 8x8: 3 rows each
        rows_per_player = 2 if self.BOARD_SIZE == 6 else 3
        
        # Place player 1 pieces (red) on top rows
        for row in range(rows_per_player):
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1
        
        # Place player 2 pieces (black) on bottom rows
        for row in range(self.BOARD_SIZE - rows_per_player, self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1
    
    def get_valid_moves(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all valid moves for a player.
        Returns list of ((from_row, from_col), (to_row, to_col)) tuples.
        Prioritizes captures if available.
        """
        moves = []
        captures = []
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row, col]
                if piece * player > 0:  # This is the player's piece
                    piece_moves, piece_captures = self._get_piece_moves(row, col, player, piece)
                    moves.extend(piece_moves)
                    captures.extend(piece_captures)
        
        # If captures are available, only return captures
        if captures:
            return captures
        return moves
    
    def _get_piece_moves(self, row: int, col: int, player: int, piece: int) -> Tuple[List, List]:
        """Get moves for a specific piece."""
        moves = []
        captures = []
        
        is_king = abs(piece) == 2
        directions = []
        
        if piece > 0:  # Player 1 (red) - moves down
            directions.append((1, -1))  # Down-left
            directions.append((1, 1))   # Down-right
        elif piece < 0:  # Player 2 (black) - moves up
            directions.append((-1, -1))  # Up-left
            directions.append((-1, 1))    # Up-right
        
        if is_king:
            # Kings can move in all diagonal directions
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Check for captures first
        for dr, dc in directions:
            capture_moves = self._get_capture_sequence(row, col, dr, dc, player, piece, [])
            captures.extend(capture_moves)
        
        # If no captures, check regular moves
        if not captures:
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if self._is_valid_position(new_row, new_col) and self.board[new_row, new_col] == 0:
                    moves.append(((row, col), (new_row, new_col)))
        
        return moves, captures
    
    def _get_capture_sequence(self, row: int, col: int, dr: int, dc: int, 
                              player: int, piece: int, path: List) -> List:
        """Get all possible capture sequences from a position."""
        captures = []
        is_king = abs(piece) == 2
        
        # Check if we can jump over an opponent piece
        jump_row, jump_col = row + dr, col + dc
        land_row, land_col = row + 2*dr, col + 2*dc
        
        if (self._is_valid_position(jump_row, jump_col) and 
            self._is_valid_position(land_row, land_col) and
            self.board[jump_row, jump_col] * player < 0 and  # Opponent piece
            self.board[land_row, land_col] == 0):  # Landing square is empty
            
            # Check if this position was already visited in this path
            if (land_row, land_col) in path:
                return captures
            
            # Make the capture
            new_path = path + [(row, col), (land_row, land_col)]
            
            # Check for further captures from the landing position
            further_captures = []
            for new_dr, new_dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if is_king or (new_dr * player > 0 if piece > 0 else new_dr * player < 0):
                    further = self._get_capture_sequence(land_row, land_col, new_dr, new_dc, 
                                                         player, piece, new_path)
                    further_captures.extend(further)
            
            if further_captures:
                captures.extend(further_captures)
            else:
                captures.append(((row, col), (land_row, land_col)))
        
        return captures
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is on the board."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Make a move on the board.
        Returns True if move was successful, False otherwise.
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        if not self._is_valid_position(from_row, from_col) or not self._is_valid_position(to_row, to_col):
            return False
        
        piece = self.board[from_row, from_col]
        if piece * self.current_player <= 0:
            return False  # Not the player's piece
        
        # Check if it's a capture
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        if row_diff == 2 and col_diff == 2:
            # This is a capture
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            if self.board[captured_row, captured_col] * self.current_player < 0:
                self.board[captured_row, captured_col] = 0
            else:
                return False  # Invalid capture
        
        # Move the piece
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0
        
        # Check for promotion to king
        if piece == 1 and to_row == self.BOARD_SIZE - 1:
            self.board[to_row, to_col] = 2
        elif piece == -1 and to_row == 0:
            self.board[to_row, to_col] = -2
        
        # Switch player
        self.current_player *= -1
        
        # Check for game over
        self._check_game_over()
        
        return True
    
    def _check_game_over(self):
        """Check if the game is over."""
        moves_p1 = self.get_valid_moves(1)
        moves_p2 = self.get_valid_moves(-1)
        
        pieces_p1 = np.sum((self.board > 0).astype(int))
        pieces_p2 = np.sum((self.board < 0).astype(int))
        
        if pieces_p1 == 0 or len(moves_p1) == 0:
            self.game_over = True
            self.winner = -1
        elif pieces_p2 == 0 or len(moves_p2) == 0:
            self.game_over = True
            self.winner = 1
        else:
            self.game_over = False
            self.winner = None
    
    def copy(self):
        """Create a deep copy of the board."""
        new_board = DraughtsBoard()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        return new_board
    
    def get_board_features(self) -> np.ndarray:
        """
        Extract features from the board for use in strategy evaluation.
        Returns a flattened array of board features.
        """
        # Flatten the board (board_size^2 values)
        features = self.board.flatten()
        
        # Add additional features
        piece_count_p1 = np.sum((self.board > 0).astype(int))
        piece_count_p2 = np.sum((self.board < 0).astype(int))
        king_count_p1 = np.sum((self.board == 2).astype(int))
        king_count_p2 = np.sum((self.board == -2).astype(int))
        
        # Center control (pieces in center area)
        # For 6x6: center 2x2, for 8x8: center 4x4
        if self.BOARD_SIZE == 6:
            center = self.board[2:4, 2:4]
        else:
            center = self.board[2:6, 2:6]
        center_control_p1 = np.sum((center > 0).astype(int))
        center_control_p2 = np.sum((center < 0).astype(int))
        
        # Back row protection
        back_row_p1 = np.sum((self.board[0, :] > 0).astype(int))
        back_row_p2 = np.sum((self.board[self.BOARD_SIZE - 1, :] < 0).astype(int))
        
        # Mobility (number of valid moves)
        mobility_p1 = len(self.get_valid_moves(1))
        mobility_p2 = len(self.get_valid_moves(-1))
        
        # Combine all features
        additional_features = np.array([
            piece_count_p1, piece_count_p2,
            king_count_p1, king_count_p2,
            center_control_p1, center_control_p2,
            back_row_p1, back_row_p2,
            mobility_p1, mobility_p2,
            self.current_player
        ])
        
        return np.concatenate([features, additional_features])


def random_strategy(features, valid_moves):
    """Random baseline strategy for comparison."""
    import random
    return random.randint(0, len(valid_moves) - 1) if valid_moves else 0


def play_game(strategy_p1, strategy_p2, max_moves=200, board_size=6):
    """
    Play a game between two strategies.
    
    Args:
        strategy_p1: Function that takes board features and returns move index for player 1
        strategy_p2: Function that takes board features and returns move index for player 2
        max_moves: Maximum number of moves before draw
        board_size: Size of the board (6 for simplified, 8 for standard)
    
    Returns:
        Result: 1 if player 1 wins, -1 if player 2 wins, 0 if draw
    """
    board = DraughtsBoard(board_size=board_size)
    move_count = 0
    
    while not board.game_over and move_count < max_moves:
        current_player = board.current_player
        valid_moves = board.get_valid_moves(current_player)
        
        if not valid_moves:
            board.game_over = True
            board.winner = -current_player
            break
        
        # Get board features
        features = board.get_board_features()
        
        # Select move based on strategy
        if current_player == 1:
            move_idx = strategy_p1(features, valid_moves)
        else:
            move_idx = strategy_p2(features, valid_moves)
        
        # Ensure valid move index
        move_idx = max(0, min(move_idx, len(valid_moves) - 1))
        from_pos, to_pos = valid_moves[move_idx]
        
        board.make_move(from_pos, to_pos)
        move_count += 1
    
    if move_count >= max_moves:
        # Draw - evaluate by piece count
        pieces_p1 = np.sum((board.board > 0).astype(int))
        pieces_p2 = np.sum((board.board < 0).astype(int))
        if pieces_p1 > pieces_p2:
            return 1
        elif pieces_p2 > pieces_p1:
            return -1
        else:
            return 0
    
    return board.winner if board.winner else 0

