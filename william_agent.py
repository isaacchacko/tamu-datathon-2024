# test_cases_1.py (repeat this structure for test_cases_2.py, test_cases_3.py, and test_cases_4.py)
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(filename='test_cases_1.log',
                    level=logging.INFO, format='%(asctime)s - %(message)s')


def evaluate_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            f"{func.__name__} - Elapsed time: {elapsed_time:.6f} seconds")
        return result
    return wrapper


# example function for static checks
@evaluate_runtime
def static_weight(game,
                  board,
                  turn_count,
                  attempt_number,
                  lastFitness):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    output = np.zeros((8, 8))

    # add your code here

    return output

# example function for pattern checks


@evaluate_runtime
def get_initial_fitness(game,
                        board,
                        turn_count,
                        attempt_number,
                        lastFitness):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    fitness = 0
    output = np.zeros((8, 8))

    # add your code here

    return fitness, output


def get_hot(heatmap):
    # Ensure the input is a NumPy array
    heatmap = np.array(heatmap)

    # Check if the input is an 8x8 array
    if heatmap.shape != (8, 8):
        raise ValueError("Input heatmap must be an 8x8 array")

    # Flatten the array and get the indices of the top 7 values
    flat_indices = np.argpartition(heatmap.ravel(), -7)[-7:]

    # Sort the indices based on their values in descending order
    sorted_indices = flat_indices[np.argsort(
        heatmap.ravel()[flat_indices])[::-1]]

    # Convert flat indices to 2D coordinates
    top_7_locations = [(index // 8, index % 8) for index in sorted_indices]

    return top_7_locations


def get_coldest(heatmap, game):
    """
    Find the coldest cell that contains an ally piece.

    Parameters:
    heatmap (np.array): 8x8 numpy array representing the heatmap
    game (Game): The current game state

    Returns:
    tuple: (y, x) coordinates of the coldest cell with an ally piece, or None if no such cell exists
    """
    # Ensure the heatmap is a NumPy array
    heatmap = np.array(heatmap)

    # Check if the input is an 8x8 array
    if heatmap.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("Input heatmap must be an 8x8 array")

    # Create a mask for ally pieces
    ally_mask = game.board == game.current_player

    # Apply the mask to the heatmap
    masked_heatmap = np.where(ally_mask, heatmap, np.inf)

    # Find the minimum value in the masked heatmap
    min_value = np.min(masked_heatmap)

    # If there are no ally pieces, return None
    if np.isinf(min_value):
        return None

    # Find the coordinates of the minimum value
    y, x = np.unravel_index(np.argmin(masked_heatmap), masked_heatmap.shape)

    return (y, x)


def get_possible_heatmap_moves(game, coldest_cell, target_cells):
    """
    Get all possible moves that either PLACE into or MOVE into the given target cells.

    Parameters:
    game (Game): The current game state
    coldest_cell (tuple): (y, x) coordinates of the cell to move from (for MOVE actions)
    target_cells (list): List of (y, x) coordinates of target cells to move to or place in

    Returns:
    list: List of valid moves (each move is a list of integers)
    """
    possible_moves = []
    current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces

    # If we haven't placed all pieces, consider PLACE actions
    if current_pieces < NUM_PIECES:
        for y, x in target_cells:
            if game.is_valid_placement(y, x):
                possible_moves.append([y, x])  # PLACE move

    # Consider MOVE actions
    if current_pieces == NUM_PIECES:
        cy, cx = coldest_cell
        if game.board[cy][cx] == game.current_player:
            for y, x in target_cells:
                if game.is_valid_move(cy, cx, y, x):
                    possible_moves.append([cy, cx, y, x])  # MOVE move

    return possible_moves

def evaluate_piece_count(board, plater):
    piece_count = sum(1 for row in board for cell in row if cell == plater)
    return piece_count
######### WILLIAM THINGS ##################

@evaluate_runtime
def calculate_board_score(board, player):
    EMPTY = 0  # Assume EMPTY is represented as 0
    enemy = 3 - player  # Assuming players are represented as 1 and 2
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, two diagonals
    fitness = 0

    def check_line(y, x, dy, dx, length):
        line = [board[(y + i*dy) % len(board)][(x + i*dx) % len(board)] for i in range(length)]
        return line

    for y in range(len(board)):
        for x in range(len(board)):
            if board[y][x] == EMPTY:
                for dy, dx in directions:
                    # Check for ally conditions
                    line = check_line(y, x, dy, dx, 5)
                    if line[2] == EMPTY:
                        if line[1] == player and line[3] == player:
                            if line[0] == EMPTY and line[4] == EMPTY:
                                fitness += 10  # Condition 1: Two ally tiles with open spaces on both sides
                            else:
                                fitness += 7   # Condition 5: Two ally tiles with a space between
                    elif line.count(player) >= 2 and line[0] == EMPTY and line[4] == EMPTY:
                        fitness += 10  # Condition 1: Two or more ally tiles with open spaces on both sides

                    # Check for enemy conditions
                    if line.count(enemy) >= 2:
                        if line[0] == EMPTY and line[4] == EMPTY:
                            fitness += 10  # Condition 2: Two or more enemy tiles with open spaces
                        elif line[2] == EMPTY and line[1] == enemy and line[3] == enemy:
                            fitness = 0  # Condition 3: Two enemy tiles with a space between

                    # Check for 3 in a row
                    if line[1:4].count(player) == 3:
                        fitness += 20  # Condition 4: 3 in a row for ally

            elif board[y][x] != EMPTY:
                fitness = 0  # Condition 6: Space is already occupied

    return fitness



def get_best_move(board, player):
    best_score = float('-inf')
    best_move = None

    for y in range(len(board)):
        for x in range(len(board)):
            if board[y][x] == EMPTY:
                board[y][x] = player
                score = calculate_board_score(board, player)
                board[y][x] = EMPTY

                if score > best_score:
                    best_score = score
                    best_move = (y, x)

    return best_move, best_score


def get_fitness(game,
                turn_count,
                move):  # move is (y, x)

    # create the new game state
    game_copy = Game.from_dict(game.to_dict())
    if len(move) == 2:
        game_copy.place_checker(move[0], move[1])
    else:
        game_copy.move_checker(move[0], move[1], move[2], move[3])

    # useful variables
    board = game_copy.board
    p1_pieces = game_copy.p1_pieces
    p2_pieces = game_copy.p2_pieces
    current_player = game_copy.current_player
    if current_player == 1:
        current_player_pieces = p1_pieces
    else:
        current_player_pieces = p2_pieces
    turn_count += 1
    

    fitness = 0
    # add your fitness check functions here
    fitness += allies_two_in_a_row(board, current_player * -1)
    fitness += enemy_two_in_a_row(board, current_player * -1)
    fitness += enemy_close_to_two(board, current_player * -1)
    
    
        
    return fitness


