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
            # Filter to only include moves that start from this piece's position
            # (continuation moves from intermediate positions will be handled automatically in make_move)
            for move in capture_moves:
                move_from, move_to = move
                if move_from == (row, col):
                    captures.append(move)
        
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
            
            # Always include the initial jump
            initial_jump = ((row, col), (land_row, land_col))
            
            # Check for further captures from the landing position
            further_captures = []
            for new_dr, new_dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if is_king or (new_dr * player > 0 if piece > 0 else new_dr * player < 0):
                    further = self._get_capture_sequence(land_row, land_col, new_dr, new_dc, 
                                                         player, piece, new_path)
                    further_captures.extend(further)
            
            # Always include the initial jump (make_move will handle continuation automatically)
            captures.append(initial_jump)
            # Also include further captures for completeness (they'll be filtered in _get_piece_moves)
            if further_captures:
                captures.extend(further_captures)
        
        return captures
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is on the board."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Make a move on the board.
        Returns True if move was successful, False otherwise.
        Automatically continues multi-jump sequences per English draughts rules.
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
            # This is a capture - execute it and check for continuation
            captured_row = (from_row + to_row) // 2
            captured_col = (from_col + to_col) // 2
            if self.board[captured_row, captured_col] * self.current_player < 0:
                self.board[captured_row, captured_col] = 0
            else:
                return False  # Invalid capture
        
        # Move the piece
        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0
        
        # Check for promotion to king (before checking for continuation captures)
        if piece == 1 and to_row == self.BOARD_SIZE - 1:
            self.board[to_row, to_col] = 2
            piece = 2  # Update piece value for continuation checks
        elif piece == -1 and to_row == 0:
            self.board[to_row, to_col] = -2
            piece = -2  # Update piece value for continuation checks
        
        # If this was a capture, check for continuation captures (multi-jump)
        if row_diff == 2 and col_diff == 2:
            # Check if there are more captures available from the landing position
            # Note: current_player hasn't switched yet, so it's still the capturing player
            # Get the actual piece value from the board (in case it was promoted)
            actual_piece = self.board[to_row, to_col]
            continuation_captures = self._get_continuation_captures(to_row, to_col, self.current_player, actual_piece)
            
            if continuation_captures:
                # Continue with the longest capture sequence (English draughts rule)
                # Execute all possible continuation captures automatically
                current_pos = (to_row, to_col)
                current_piece = actual_piece  # Use the actual piece value (may have been promoted)
                visited_positions = [(from_row, from_col), current_pos]  # Track visited to avoid loops
                
                while True:
                    # Get all possible continuation captures from current position
                    # Use current_piece which may have been updated if promotion occurred
                    next_captures = self._get_continuation_captures(
                        current_pos[0], current_pos[1], self.current_player, current_piece
                    )
                    
                    if not next_captures:
                        break  # No more captures available
                    
                    # Find a valid continuation capture (avoid positions we've already visited)
                    best_capture = None
                    
                    for cap_from, cap_to in next_captures:
                        # Avoid positions we've already visited in this sequence
                        if cap_to not in visited_positions:
                            best_capture = (cap_from, cap_to)
                            break  # Take the first valid continuation
                    
                    if best_capture is None:
                        # No valid continuation (all lead to visited positions)
                        break
                    
                    # Execute the continuation capture
                    cap_from_pos, cap_to_pos = best_capture
                    cap_from_row, cap_from_col = cap_from_pos
                    cap_to_row, cap_to_col = cap_to_pos
                    
                    # Capture the piece
                    cap_captured_row = (cap_from_row + cap_to_row) // 2
                    cap_captured_col = (cap_from_col + cap_to_col) // 2
                    self.board[cap_captured_row, cap_captured_col] = 0
                    
                    # Move the piece
                    self.board[cap_to_row, cap_to_col] = current_piece
                    self.board[cap_from_row, cap_from_col] = 0
                    
                    # Check for promotion to king during continuation
                    if current_piece == 1 and cap_to_row == self.BOARD_SIZE - 1:
                        self.board[cap_to_row, cap_to_col] = 2
                        current_piece = 2
                    elif current_piece == -1 and cap_to_row == 0:
                        self.board[cap_to_row, cap_to_col] = -2
                        current_piece = -2
                    
                    # Update position and visited list
                    current_pos = (cap_to_row, cap_to_col)
                    visited_positions.append(current_pos)
        
        # Switch player (only after all captures in sequence are complete)
        self.current_player *= -1
        
        # Check for game over
        self._check_game_over()
        
        return True
    
    def _get_continuation_captures(self, row: int, col: int, player: int, piece: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all possible capture moves from a specific position (for continuation after a jump)."""
        captures = []
        is_king = abs(piece) == 2
        
        # Determine valid directions based on piece type
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if piece > 0:  # Player 1 (red) - moves down
                directions = [(1, -1), (1, 1)]
            else:  # Player 2 (black) - moves up
                directions = [(-1, -1), (-1, 1)]
        
        # Check each direction for possible captures
        for dr, dc in directions:
            jump_row, jump_col = row + dr, col + dc
            land_row, land_col = row + 2*dr, col + 2*dc
            
            if (self._is_valid_position(jump_row, jump_col) and 
                self._is_valid_position(land_row, land_col) and
                self.board[jump_row, jump_col] * player < 0 and  # Opponent piece
                self.board[land_row, land_col] == 0):  # Landing square is empty
                captures.append(((row, col), (land_row, land_col)))
        
        return captures
    
    def _get_capture_sequence_length(self, row: int, col: int, player: int, piece: int, visited: List) -> int:
        """Get the length of the longest capture sequence from a position (recursive helper)."""
        max_length = 0
        is_king = abs(piece) == 2
        
        # Determine valid directions
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if piece > 0:
                directions = [(1, -1), (1, 1)]
            else:
                directions = [(-1, -1), (-1, 1)]
        
        for dr, dc in directions:
            jump_row, jump_col = row + dr, col + dc
            land_row, land_col = row + 2*dr, col + 2*dc
            
            if (self._is_valid_position(jump_row, jump_col) and 
                self._is_valid_position(land_row, land_col) and
                self.board[jump_row, jump_col] * player < 0 and
                self.board[land_row, land_col] == 0 and
                (land_row, land_col) not in visited):
                
                # Recursively check continuation
                length = 1 + self._get_capture_sequence_length(
                    land_row, land_col, player, piece, visited + [(land_row, land_col)]
                )
                max_length = max(max_length, length)
        
        return max_length
    
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
        new_board = DraughtsBoard(board_size=self.BOARD_SIZE)
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

