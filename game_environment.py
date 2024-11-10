import numpy as np
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

def three_in_a_row_opportunities(board, player):
    opportunities = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
    
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            for dr, dc in directions:
                opportunity = check_direction(board, player, row, col, dr, dc)
                opportunities += opportunity
    
    return opportunities

def check_direction(board, player, row, col, dr, dc):
    count = 0
    empty = 0
    
    for i in range(3):
        r, c = _torus(row + i*dr, col + i*dc)
        if board[r][c] == player:
            count += 1
        elif board[r][c] == EMPTY:
            empty += 1
        else:
            return 0  # Opponent's piece, no opportunity here
    
    # Return 1 if there are two player pieces and one empty space
    return 1 if count == 2 and empty == 1 else 0

def get_reward(game, player):
    winner = game.check_winner()
    if winner == player:
        return 1.0  # Win
    elif winner == -player:
        return -1.0  # Loss
    elif winner == EMPTY:
        reward = 0
        
        # Reward for potential winning positions
        reward += 0.1 * three_in_a_row_opportunities(game.board, player)
        
        # Penalty for opponent's potential winning positions
        reward -= 0.1 * three_in_a_row_opportunities(game.board, -player)
        
        # Add other reward components here
        
        return reward
    return 0.0

# Ensure _torus is defined here or imported from PushBattle if needed
def _torus(r, c):
    return r % BOARD_SIZE, c % BOARD_SIZE