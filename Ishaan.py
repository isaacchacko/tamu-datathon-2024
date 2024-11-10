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
def Ishaan_heatmap(game,
                  board,
                  turn_count,
                  attempt_number,
                  lastFitness):
    
    # useful variables
    current_player = game.current_player
    opponent_player = -current_player  # Opponent's pieces
    p1_tiles = game.p1_tiles
    p2_tiles = game.p2_tiles

    # Initialize an 8x8 output array for the heatmap with all cells starting at zero
    output = np.zeros((8, 8))

    # Set cells with pieces (ally or enemy) to 0 in the heatmap
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != EMPTY:
                output[y][x] = 0  # Cell with a piece is set to 0

    # Determine clustering incentives based on fitness score (only if turn_count is not 1 or 2)
    if turn_count != 1 and turn_count != 2:
        ally_incentive, enemy_incentive = (1, 3) if lastFitness < 13 else (3, 1)
    else:
        ally_incentive, enemy_incentive = 0, 0  # Skip clustering incentive for early turns

    # Apply clustering incentives to empty cells based on adjacency to allies or enemies
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == EMPTY:
                # Count ally and enemy neighbors in the surrounding cells (8 directions)
                ally_neighbors = 0
                enemy_neighbors = 0

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Skip the cell itself

                        ny, nx = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE  # Wrap around (toroidal)
                        if board[ny][nx] == current_player:
                            ally_neighbors += 1
                        elif board[ny][nx] == opponent_player:
                            enemy_neighbors += 1

                # Apply clustering incentive based on number of adjacent ally/enemy tiles
                output[y][x] += (ally_neighbors * ally_incentive) + (enemy_neighbors * enemy_incentive)

    # Apply additional weights for lines of two adjacent pieces with a gap
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]  # 8 directions

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            # Check for two pieces with a gap in all 8 directions
            for dy, dx in directions:
                first_y, first_x = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE
                gap_y, gap_x = (first_y + dy) % BOARD_SIZE, (first_x + dx) % BOARD_SIZE

                # Check if we have two pieces with a gap in the middle
                if board[y][x] == current_player and board[gap_y][gap_x] == current_player and board[first_y][first_x] == EMPTY:
                    # Add +10 to the gap cell for two ally pieces with a gap
                    output[first_y][first_x] += 10
                elif board[y][x] == opponent_player and board[gap_y][gap_x] == opponent_player and board[first_y][first_x] == EMPTY:
                    # Add +4 to the gap cell for two enemy pieces with a gap
                    output[first_y][first_x] += 4
    
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            # Check for lines of two adjacent pieces in each of the 8 directions
            for dy, dx in directions:
                next_y, next_x = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE

                # Check if there is a line of two adjacent pieces
                if board[y][x] == current_player and board[next_y][next_x] == current_player:
                    # Add +10 to extend the ally line in both directions
                    extend_y1, extend_x1 = (y - dy) % BOARD_SIZE, (x - dx) % BOARD_SIZE
                    extend_y2, extend_x2 = (next_y + dy) % BOARD_SIZE, (next_x + dx) % BOARD_SIZE
                    if board[extend_y1][extend_x1] == EMPTY:
                        output[extend_y1][extend_x1] += 10
                    if board[extend_y2][extend_x2] == EMPTY:
                        output[extend_y2][extend_x2] += 10

                elif board[y][x] == opponent_player and board[next_y][next_x] == opponent_player:
                    # Apply +3 to cells perpendicular to the enemy line
                    perpendicular_moves = [(dy, -dx), (-dy, dx)]  # Perpendicular directions
                    for pdy, pdx in perpendicular_moves:
                        perp_y1, perp_x1 = (y + pdy) % BOARD_SIZE, (x + pdx) % BOARD_SIZE
                        perp_y2, perp_x2 = (next_y + pdy) % BOARD_SIZE, (next_x + pdx) % BOARD_SIZE
                        if board[perp_y1][perp_x1] == EMPTY:
                            output[perp_y1][perp_x1] += 3
                        if board[perp_y2][perp_x2] == EMPTY:
                            output[perp_y2][perp_x2] += 3
    # Apply additional weights based on lastFitness and pairs in rows, columns, or diagonals
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == EMPTY:
                continue  # Only look at occupied cells

            # Determine piece type and incentives based on lastFitness
            if lastFitness >= 13:
                ally_bonus, enemy_bonus = (3, 1)
            else:
                ally_bonus, enemy_bonus = (1, 3)

            # Check for rows, columns, and diagonals
            for dy, dx in directions[:4]:  # Limit to row/column/diagonal (avoid checking both directions redundantly)
                adj_y1, adj_x1 = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE
                adj_y2, adj_x2 = (y - dy) % BOARD_SIZE, (x - dx) % BOARD_SIZE

                if board[y][x] == current_player:
                    if board[adj_y1][adj_x1] == current_player or board[adj_y2][adj_x2] == current_player:
                        output[y][x] += ally_bonus  # Add to the entire row/column/diagonal

                elif board[y][x] == opponent_player:
                    if board[adj_y1][adj_x1] == opponent_player or board[adj_y2][adj_x2] == opponent_player:
                        output[y][x] += enemy_bonus  # Add to the entire row/column/diagonal

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
    board = game.board
    p1_pieces = game.p1_pieces
    p2_pieces = game.p2_pieces
    current_player = game.current_player * -1
    turn_count += 1

    fitness = 0
    # add your fitness check functions here

    return fitness
