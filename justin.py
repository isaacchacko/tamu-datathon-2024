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

################### JUSTIN'S SECTION ##########################

@evaluate_runtime
def evaluate_piece_count(current_player_pieces):
    return current_player_pieces

@evaluate_runtime
def evaluate_defensive_formation(board, player):
    defensive_score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]

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
                    victory_score -= check_line(line, opponent) * 0.8  # Slightly less weight for opponent's threats

    return victory_score

@evaluate_runtime
def evaluate_opponent_cluster_disruption(board, opponent):
    disruption_score = 0
    size = len(board)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

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
                                new_cluster_size = get_cluster_size(push_r, push_c)
                                # Revert the move
                                board[nr][nc], board[push_r][push_c] = opponent, None
                                
                                if new_cluster_size < original_cluster_size:
                                    disruption_score += 4  # Apply weight for disrupting clusters

    return disruption_score

@evaluate_runtime
def evaluate_potential_win_pathways(board, player):
    pathway_score = 0
    size = len(board)  # Assuming a square board
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal

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
    fitness += evaluate_distance_to_victory(board, current_player, current_player * -1)
    fitness += evaluate_opponent_cluster_disruption(board, current_player * -1)
    fitness +=  evaluate_potential_win_pathways(board, current_player)

    return fitness
