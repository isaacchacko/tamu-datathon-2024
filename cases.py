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

###### FITNESS STUFF ######


@evaluate_runtime
def evaluate_piece_count(current_player_pieces):
    return current_player_pieces


@evaluate_runtime
def evaluate_defensive_formation(board, player):
    defensive_score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1),
                  (-1, 0), (0, -1), (-1, -1), (-1, 1)]

    for r in range(8):
        for c in range(8):
            if board[r][c] == player:
                backed_up = 0
                for dr, dc in directions:
                    nr = (r + dr) % 8  # Wrap around rows
                    nc = (c + dc) % 8  # Wrap around columns
                    if board[nr][nc] == player:
                        backed_up += 1

                if backed_up >= 2:
                    defensive_score += 3.0
                if backed_up >= 3:
                    defensive_score += 2.0  # Additional bonus for extra support

    return defensive_score * 2.5  # Applying the weight


@evaluate_runtime
def evaluate_distance_to_victory(board, player, opponent):
    victory_score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    size = len(board)  # Assuming the board is square (8x8)

    def check_line(line, player):
        player_count = line.count(player)
        empty_count = line.count(None)
        if player_count + empty_count == 3:
            if player_count == 2:
                return 10  # One move away from victory
            elif player_count == 1:
                return 5   # Two moves away from victory
            elif player_count == 0:
                return 1   # Three moves away (potential future line)
        return 0

    for r in range(size):
        for c in range(size):
            for dr, dc in directions:
                line = []
                for i in range(3):
                    nr = (r + i*dr) % size  # Wrap around rows
                    nc = (c + i*dc) % size  # Wrap around columns
                    line.append(board[nr][nc])
                if len(line) == 3:
                    victory_score += check_line(line, player)
                    # Slightly less weight for opponent's threats
                    victory_score -= check_line(line, opponent) * 0.8

    return victory_score


@evaluate_runtime
def evaluate_opponent_cluster_disruption(board, opponent):
    disruption_score = 0
    size = len(board)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1),
                  (0, -1), (-1, 0), (-1, -1), (-1, 1)]

    def is_opponent(r, c):
        return board[r][c] == opponent

    def get_cluster_size(r, c):
        visited = set()
        stack = [(r, c)]
        cluster_size = 0
        while stack:
            current_r, current_c = stack.pop()
            if (current_r, current_c) not in visited:
                visited.add((current_r, current_c))
                cluster_size += 1
                for dr, dc in directions:
                    nr = (current_r + dr) % size
                    nc = (current_c + dc) % size
                    if is_opponent(nr, nc) and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return cluster_size

    # Explore the board for potential moves that disrupt opponent clusters
    for r in range(size):
        for c in range(size):
            if board[r][c] is None:  # Empty cell where we can potentially move
                for dr, dc in directions:
                    nr = (r + dr) % size
                    nc = (c + dc) % size
                    if is_opponent(nr, nc):
                        original_cluster_size = get_cluster_size(nr, nc)
                        if original_cluster_size >= 2:
                            # Simulate pushing the opponent piece
                            push_r = (nr + dr) % size
                            push_c = (nc + dc) % size
                            if board[push_r][push_c] is None:  # Can push
                                # Temporarily move the piece
                                board[nr][nc], board[push_r][push_c] = None, opponent
                                new_cluster_size = get_cluster_size(
                                    push_r, push_c)
                                # Revert the move
                                board[nr][nc], board[push_r][push_c] = opponent, None

                                if new_cluster_size < original_cluster_size:
                                    disruption_score += 4  # Apply weight for disrupting clusters

    return disruption_score


@evaluate_runtime
def evaluate_potential_win_pathways(board, player):
    pathway_score = 0
    size = len(board)  # Assuming a square board
    # Horizontal, vertical, diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def is_player_piece(r, c):
        return board[r][c] == player

    def is_empty(r, c):
        return board[r][c] is None

    # Check all empty spaces for potential win pathways
    for r in range(size):
        for c in range(size):
            if is_empty(r, c):  # Only check empty cells
                for dr, dc in directions:
                    count = 0
                    empty_count = 0
                    # Check in both directions
                    for i in range(-2, 3):  # Check 2 spaces before and after
                        if i == 0:
                            continue  # Skip the current cell
                        nr = (r + i * dr) % size  # Wrap around rows
                        nc = (c + i * dc) % size  # Wrap around columns
                        if is_player_piece(nr, nc):
                            count += 1
                        elif is_empty(nr, nc):
                            empty_count += 1
                        else:
                            break  # Blocked by opponent's piece

                    # Evaluate the potential pathway
                    if count + empty_count >= 2:
                        if count == 2:
                            pathway_score += 3  # Higher score for two pieces already aligned
                        elif count == 1:
                            pathway_score += 2  # Standard score for one piece in the pathway
                        else:
                            pathway_score += 1  # Small score for potential future alignment

    return pathway_score


def calculate_board_score(board, player):
    EMPTY = 0  # Assume EMPTY is represented as 0
    enemy = 3 - player  # Assuming players are represented as 1 and 2
    # horizontal, vertical, two diagonals
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    fitness = 0

    def check_line(y, x, dy, dx, length):
        line = [board[(y + i*dy) % len(board)][(x + i*dx) %
                                               len(board)] for i in range(length)]
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


@evaluate_runtime
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
    fitness += evaluate_piece_count(current_player_pieces)
    fitness += evaluate_defensive_formation(board, current_player)
    fitness += evaluate_distance_to_victory(board,
                                            current_player, current_player * -1)
    fitness += evaluate_opponent_cluster_disruption(board, current_player * -1)
    fitness += evaluate_potential_win_pathways(board, current_player)
    fitness += get_best_move(board, current_player)

    return fitness

# ISHAAN BANSAL

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
    p1_tiles = game.p1_pieces
    p2_tiles = game.p2_pieces

    # Initialize an 8x8 output array for the heatmap with all cells starting at zero
    output = np.zeros((8, 8))

    # Set cells with pieces (ally or enemy) to 0 in the heatmap
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != EMPTY:
                output[y][x] = 0  # Cell with a piece is set to 0

    # Determine clustering incentives based on fitness score (only if turn_count is not 1 or 2)
    if turn_count != 1 and turn_count != 2:
        ally_incentive, enemy_incentive = (
            1, 3) if lastFitness < 10 else (3, 1)
    else:
        # Skip clustering incentive for early turns
        ally_incentive, enemy_incentive = 0, 0

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

                        # Wrap around (toroidal)
                        ny, nx = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE
                        if board[ny][nx] == current_player:
                            ally_neighbors += 1
                        elif board[ny][nx] == opponent_player:
                            enemy_neighbors += 1

                # Apply clustering incentive based on number of adjacent ally/enemy tiles
                output[y][x] += (ally_neighbors * ally_incentive) + \
                    (enemy_neighbors * enemy_incentive)

    # Apply additional weights for lines of two adjacent pieces with a gap
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
                  (1, 1), (-1, 1), (1, -1)]  # 8 directions

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            # Check for two pieces with a gap in all 8 directions
            for dy, dx in directions:
                first_y, first_x = (y + dy) % BOARD_SIZE, (x + dx) % BOARD_SIZE
                gap_y, gap_x = (
                    first_y + dy) % BOARD_SIZE, (first_x + dx) % BOARD_SIZE

                # Check if we have two pieces with a gap in the middle
                if board[y][x] == current_player and board[gap_y][gap_x] == current_player and board[first_y][first_x] == EMPTY:
                    # Add +10 to the gap cell for two ally pieces with a gap
                    output[first_y][first_x] += 10
                elif board[y][x] == opponent_player and board[gap_y][gap_x] == opponent_player and board[first_y][first_x] == EMPTY:
                    # Add +4 to the gap cell for two enemy pieces with a gap
                    output[first_y][first_x] += 4

    return output
# ISAAC CHACKO


@evaluate_runtime
def RadialConvolution(game,
                      board,
                      turn_count,
                      attempt_number,
                      lastFitness):

    # useful variables
    current_player = game.current_player
    p1_tiles = game.p1_pieces
    p2_tiles = game.p2_pieces

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


def combine_heatmaps(heatmap_functions,
                     game,
                     board,
                     turn_count,
                     attempt_number,
                     lastFitness):
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
        heatmap = func(game,
                       board,
                       turn_count,
                       attempt_number,
                       lastFitness)

        # Check if the generated heatmap is a valid 8x8 NumPy array
        if not isinstance(heatmap, np.ndarray) or heatmap.shape != (8, 8):
            raise ValueError(
                f"Function {func.__name__} did not return a valid 8x8 NumPy array")

        # Add the current heatmap to the result
        result += heatmap

    return result


def run_cases(game,
              board,
              turn_count,
              attempt_number,
              lastFitness):

    heatmap_array = combine_heatmaps(
        [Ishaan_heatmap, RadialConvolution], game, board, turn_count, attempt_number, lastFitness)

    # get the shrinked list of heatmap tiles
    final_heatmap = get_hot(heatmap_array)

    # get the shittiest tile on the board
    shitter = get_coldest(heatmap_array, game)

    # generate moves for sim
    possible_moves = get_possible_heatmap_moves(game, shitter, final_heatmap)

    move_fitness_dict = {}
    for possible_move in possible_moves:
        move_fitness_dict[possible_move] = get_fitness(game,
                                                       turn_count,
                                                       possible_move)

    return move_fitness_dict


def get_best_move_and_fitness(move_fitness_dict):
    """
    Get the move with the highest fitness value and its corresponding fitness from a dictionary of moves and their fitness values.

    Parameters:
    move_fitness_dict (dict): A dictionary where keys are moves and values are their fitness scores.

    Returns:
    tuple: A tuple containing (best_move, best_fitness), or (None, None) if the dictionary is empty.
    """
    if not move_fitness_dict:
        return None, None  # Return None, None if the dictionary is empty

    # Find the move with the maximum fitness value
    best_move = max(move_fitness_dict, key=move_fitness_dict.get)
    best_fitness = move_fitness_dict[best_move]

    return best_move, best_fitness
