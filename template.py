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
                  wentFirst):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    output = np.zeros((8, 8))

    # add your code here

    return output

# example function for pattern checks


@evaluate_runtime
def initial_fitness(game,
                    board,
                    turn_count,
                    attempt_number,
                    wentFirst):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    output = np.zeros((8, 8))

    # add your code here

    return output


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


def evaluate_move(game,
                  board,
                  turn_count,
                  attempt_number,
                  wentFirst,
                  move):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    # Apply the move to a copy of the game
    game_copy = Game.from_dict(game.to_dict())
    if len(move) == 2:
        game_copy.place_checker(move[0], move[1])
    else:
        game_copy.move_checker(move[0], move[1], move[2], move[3])

    fitness = 0

    # add your pattern checks here

    return fitness
