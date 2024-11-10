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
    current_player = game_copy.current_player * -1
    turn_count += 1

    fitness = 0
    # add your fitness check functions here

    return fitness


def combine_heatmaps(heatmap_functions, game):
    """
    Combine multiple heatmaps generated by different functions.

    Parameters:
    heatmap_functions (list): List of functions that generate 8x8 NumPy arrays
    game (Game): The current game state

    Returns:
    numpy.ndarray: The combined 8x8 heatmap
    """
    # Initialize the result array with zeros
    result = np.zeros((8, 8), dtype=float)

    # Iterate through each heatmap function
    for func in heatmap_functions:
        # Generate the heatmap using the current function
        heatmap = func(game)

        # Check if the generated heatmap is a valid 8x8 NumPy array
        if not isinstance(heatmap, np.ndarray) or heatmap.shape != (8, 8):
            raise ValueError(
                f"Function {func.__name__} did not return a valid 8x8 NumPy array")

        # Add the current heatmap to the result
        result += heatmap

    return result


def combine_heatmaps_and_fitness(evaluation_functions, game):
    """
    Combine multiple heatmaps and fitness values generated by different evaluation functions.

    Parameters:
    evaluation_functions (list): List of functions that generate fitness values and 8x8 NumPy array heatmaps
    game (Game): The current game state

    Returns:
    tuple: (combined_heatmap, total_fitness)
        combined_heatmap (numpy.ndarray): The combined 8x8 heatmap
        total_fitness (float): The sum of all fitness values
    """
    # Initialize the result array with zeros and total fitness
    combined_heatmap = np.zeros((8, 8), dtype=float)
    total_fitness = 0.0

    # Iterate through each evaluation function
    for func in evaluation_functions:
        # Generate the fitness and heatmap using the current function
        fitness, heatmap = func(game)

        # Check if the generated heatmap is a valid 8x8 NumPy array
        if not isinstance(heatmap, np.ndarray) or heatmap.shape != (8, 8):
            raise ValueError(
                f"Function {func.__name__} did not return a valid 8x8 NumPy array heatmap")

        # Add the current heatmap to the combined heatmap
        combined_heatmap += heatmap

        # Add the fitness value to the total fitness
        total_fitness += fitness

    return combined_heatmap, total_fitness


def run_cases(game,
              board,
              turn_count,
              attempt_number,
              lastFitness):

    heatmap_array = combine_heatmaps( < insert heatmap functions here > )
    combined_heatmap, total_initial_fitness = combine_heatmaps_and_fitness( < insert initial fitness functions here > )

    # sum the heatmaps together
    total_heatmap_array = heatmap_array + combine_heatmap

    # get the shrinked list of heatmap tiles
    final_heatmap = get_hot(total_heatmap_array)

    # get the shittiest tile on the board
    shitter = get_coldest(total_heatmap_array, game)

    # generate moves for sim
    possible_moves = get_possible_heatmap_moves(game, shitter, final_heatmap)

    move_fitness_dict = {}
    for possible_move in possible_moves:
        move_fitness_dict[possible_move] = get_fitness(game,
                                                       turn_count,
                                                       possible_move)

    return move_fitness_dict
