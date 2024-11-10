
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


@evaluate_runtime
def RadialConvolution(game,
                      board,
                      turn_count,
                      attempt_number,
                      lastFitness):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    output = np.zeros((8, 8))

    """
    Count enemy tiles in a 5x5 window traversing an 8x8 grid with wrap-around.
    
    Parameters:
    grid (numpy.ndarray): 8x8 numpy array representing the game board
    player (int): Current player (1 or -1)
    
    Returns:
    numpy.ndarray: 8x8 numpy array with counts of enemy tiles in each 5x5 window
    """
    enemy = -current_player
    output = np.zeros((8, 8), dtype=int)

    for center_y in range(8):
        for center_x in range(8):
            count = 0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    y = (center_y + i) % 8
                    x = (center_x + j) % 8
                    if grid[y, x] == enemy:
                        count += 1

            output[center_y, center_x] = count

    return output
