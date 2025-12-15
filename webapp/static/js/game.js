/**
 * EvoDraughts Game Client
 * Handles board rendering, game logic, and API communication
 */

class DraughtsGame {
    constructor() {
        this.canvas = document.getElementById('game-board');
        this.ctx = this.canvas.getContext('2d');
        this.board = null;
        this.boardSize = 6;
        this.cellSize = 0;
        this.selectedCell = null;
        this.validMoves = [];
        this.currentPlayer = 1; // 1 = human (red), -1 = AI (black)
        this.gameOver = false;
        
        // Bind methods
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        
        // UI event listeners
        document.getElementById('start-btn').addEventListener('click', () => this.startGame());
        document.getElementById('new-game-btn').addEventListener('click', () => this.showSettings());
    }
    
    showSettings() {
        document.getElementById('settings-panel').classList.remove('hidden');
        document.getElementById('game-panel').classList.add('hidden');
        this.board = null;
        this.selectedCell = null;
        this.validMoves = [];
    }
    
    async startGame() {
        const complexity = document.getElementById('complexity').value;
        const difficulty = document.getElementById('difficulty').value;
        
        // Extract board size from complexity
        this.boardSize = parseInt(complexity.split('x')[0]);
        
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/game/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    complexity: complexity,
                    difficulty: difficulty
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                alert('Error: ' + data.error);
                this.showLoading(false);
                return;
            }
            
            this.board = data.board;
            this.currentPlayer = data.current_player;
            this.validMoves = data.valid_moves;
            this.gameOver = data.game_over;
            
            // Update UI
            document.getElementById('settings-panel').classList.add('hidden');
            document.getElementById('game-panel').classList.remove('hidden');
            document.getElementById('current-difficulty').textContent = 
                difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
            
            this.updateGameInfo();
            this.resizeCanvas();
            this.drawBoard();
            
        } catch (error) {
            console.error('Error starting game:', error);
            alert('Failed to start game. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const maxSize = Math.min(600, container.clientWidth - 40);
        this.cellSize = Math.floor(maxSize / this.boardSize);
        const boardSize = this.cellSize * this.boardSize;
        
        this.canvas.width = boardSize;
        this.canvas.height = boardSize;
        this.canvas.style.width = boardSize + 'px';
        this.canvas.style.height = boardSize + 'px';
    }
    
    drawBoard() {
        const ctx = this.ctx;
        const size = this.cellSize;
        
        // Clear canvas
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw board squares
        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                const x = col * size;
                const y = row * size;
                
                // Checkerboard pattern
                const isLight = (row + col) % 2 === 0;
                ctx.fillStyle = isLight ? '#f0d9b5' : '#b58863';
                ctx.fillRect(x, y, size, size);
                
                // Highlight selected cell
                if (this.selectedCell && this.selectedCell[0] === row && this.selectedCell[1] === col) {
                    ctx.fillStyle = 'rgba(255, 255, 0, 0.4)';
                    ctx.fillRect(x, y, size, size);
                }
                
                // Highlight valid move destinations
                if (this.selectedCell) {
                    const fromRow = this.selectedCell[0];
                    const fromCol = this.selectedCell[1];
                    for (let move of this.validMoves) {
                        if (move[0][0] === fromRow && move[0][1] === fromCol) {
                            if (move[1][0] === row && move[1][1] === col) {
                                ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
                                ctx.fillRect(x, y, size, size);
                            }
                        }
                    }
                }
            }
        }
        
        // Draw pieces
        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                const piece = this.board[row][col];
                if (piece !== 0) {
                    this.drawPiece(row, col, piece);
                }
            }
        }
    }
    
    drawPiece(row, col, piece) {
        const ctx = this.ctx;
        const size = this.cellSize;
        const centerX = col * size + size / 2;
        const centerY = row * size + size / 2;
        const radius = size * 0.35;
        
        // Determine piece color
        const isKing = Math.abs(piece) === 2;
        const isRed = piece > 0;
        
        // Draw piece shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.arc(centerX + 2, centerY + 2, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw piece base
        ctx.fillStyle = isRed ? '#dc3545' : '#212529';
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw piece border
        ctx.strokeStyle = isRed ? '#a02835' : '#0d0f11';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw king crown
        if (isKing) {
            ctx.fillStyle = '#ffd700';
            ctx.font = `bold ${radius * 0.8}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('♔', centerX, centerY);
        }
    }
    
    handleCanvasClick(e) {
        if (this.gameOver || this.currentPlayer !== 1) {
            return;
        }
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const col = Math.floor(x / this.cellSize);
        const row = Math.floor(y / this.cellSize);
        
        if (row < 0 || row >= this.boardSize || col < 0 || col >= this.boardSize) {
            return;
        }
        
        // Check if clicking on a valid move destination
        if (this.selectedCell) {
            const fromRow = this.selectedCell[0];
            const fromCol = this.selectedCell[1];
            
            // Check if this is a valid move
            for (let move of this.validMoves) {
                if (move[0][0] === fromRow && move[0][1] === fromCol &&
                    move[1][0] === row && move[1][1] === col) {
                    // Make the move
                    this.makeMove([fromRow, fromCol], [row, col]);
                    return;
                }
            }
        }
        
        // Check if clicking on own piece
        const piece = this.board[row][col];
        if (piece > 0) {
            // Check if this piece has valid moves
            const hasMoves = this.validMoves.some(move => 
                move[0][0] === row && move[0][1] === col
            );
            
            if (hasMoves) {
                this.selectedCell = [row, col];
                this.drawBoard();
            }
        } else {
            // Deselect if clicking empty space
            this.selectedCell = null;
            this.drawBoard();
        }
    }
    
    async makeMove(from, to) {
        this.selectedCell = null;
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/game/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    from: from,
                    to: to
                })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                alert('Error: ' + data.error);
                this.showLoading(false);
                return;
            }
            
            this.board = data.board;
            this.currentPlayer = data.current_player;
            this.validMoves = data.valid_moves || [];
            this.gameOver = data.game_over;
            
            this.updateGameInfo();
            
            // Draw the board first (it already has the AI's move applied)
            this.drawBoard();
            
            // Show AI move animation if available
            if (data.ai_move) {
                // Add delay before showing AI move animation to make it more visible
                setTimeout(() => {
                    this.highlightAIMove(data.ai_move.from, data.ai_move.to);
                }, 800); // 0.8 second delay
            }
            
            // Check if game is over
            if (this.gameOver) {
                setTimeout(() => {
                    this.showGameOver(data.winner);
                }, 2500); // Increased delay to allow animation to complete
            }
            
        } catch (error) {
            console.error('Error making move:', error);
            alert('Failed to make move. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    highlightAIMove(from, to) {
        // Enhanced animation to clearly show AI's move
        const ctx = this.ctx;
        const size = this.cellSize;
        
        // First, highlight the source square with pulsing effect
        let pulseCount = 0;
        const maxPulses = 2;
        const pulseDuration = 400; // milliseconds per pulse
        const startTime = Date.now();
        
        const highlightSource = () => {
            const elapsed = Date.now() - startTime;
            const cycle = (elapsed % pulseDuration) / pulseDuration;
            const pulse = Math.sin(cycle * Math.PI * 2) * 0.3 + 0.5;
            
            this.drawBoard();
            
            // Draw pulsing highlight on source square (where piece was)
            ctx.fillStyle = `rgba(5, 150, 105, ${pulse})`;
            ctx.fillRect(from[1] * size, from[0] * size, size, size);
            
            pulseCount++;
            if (pulseCount < maxPulses * 30) { // ~2 seconds at 60fps
                requestAnimationFrame(highlightSource);
            } else {
                // Now show the arrow animation to destination
                this.animateMoveArrow(from, to);
            }
        };
        
        // Start highlighting source
        highlightSource();
    }
    
    animateMoveArrow(from, to) {
        const ctx = this.ctx;
        const size = this.cellSize;
        
        // Draw arrow/line animation from source to destination
        let progress = 0;
        const duration = 500; // milliseconds
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            
            // Redraw board
            this.drawBoard();
            
            // Draw animated arrow from source to destination
            const fromX = from[1] * size + size / 2;
            const fromY = from[0] * size + size / 2;
            const toX = to[1] * size + size / 2;
            const toY = to[0] * size + size / 2;
            const currentX = fromX + (toX - fromX) * easeProgress;
            const currentY = fromY + (toY - fromY) * easeProgress;
            
            // Draw line
            ctx.strokeStyle = `rgba(5, 150, 105, ${0.9})`;
            ctx.lineWidth = 5;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(fromX, fromY);
            ctx.lineTo(currentX, currentY);
            ctx.stroke();
            
            // Draw circle at current position (moving along the line)
            ctx.fillStyle = `rgba(5, 150, 105, 0.8)`;
            ctx.beginPath();
            ctx.arc(currentX, currentY, size * 0.25, 0, Math.PI * 2);
            ctx.fill();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                // Final highlight flash on destination
                this.flashDestination(to);
            }
        };
        
        animate();
    }
    
    flashDestination(to) {
        const ctx = this.ctx;
        const size = this.cellSize;
        
        const flashDuration = 400;
        const startTime = Date.now();
        
        const flash = () => {
            const elapsed = Date.now() - startTime;
            const flashProgress = Math.min(elapsed / flashDuration, 1);
            const alpha = 0.6 * (1 - flashProgress);
            
            this.drawBoard();
            // Draw pulsing highlight on destination
            ctx.fillStyle = `rgba(5, 150, 105, ${alpha})`;
            ctx.fillRect(to[1] * size, to[0] * size, size, size);
            
            if (flashProgress < 1) {
                requestAnimationFrame(flash);
            } else {
                // Final redraw - animation complete
                this.drawBoard();
            }
        };
        
        flash();
    }
    
    updateGameInfo() {
        const playerText = this.currentPlayer === 1 ? 'You (Red)' : 'AI (Black)';
        document.getElementById('current-player').textContent = playerText;
        
        // Get move count from server
        this.fetchGameState();
    }
    
    async fetchGameState() {
        try {
            const response = await fetch('/api/game/state');
            const data = await response.json();
            
            if (response.ok) {
                document.getElementById('move-count').textContent = data.move_count || 0;
            }
        } catch (error) {
            console.error('Error fetching game state:', error);
        }
    }
    
    showGameOver(winner) {
        const statusDiv = document.getElementById('game-status');
        statusDiv.classList.remove('hidden');
        
        if (winner === 1) {
            statusDiv.className = 'game-status win';
            statusDiv.textContent = 'You Win!';
        } else if (winner === -1) {
            statusDiv.className = 'game-status loss';
            statusDiv.textContent = 'GE Wins';
        } else {
            statusDiv.className = 'game-status draw';
            statusDiv.textContent = 'Draw';
        }
    }
    
    showLoading(show) {
        const loading = document.getElementById('loading');
        if (show) {
            loading.classList.remove('hidden');
        } else {
            loading.classList.add('hidden');
        }
    }
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.game = new DraughtsGame();
});

