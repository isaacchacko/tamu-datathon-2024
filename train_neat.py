# train_neat.py
import neat
import os
import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES
import pickle

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        
        # Play multiple games to get a better fitness estimate
        for _ in range(5):
            genome.fitness += play_game(net)

def play_game(net):
    game = Game()
    max_moves = 100  # Set a maximum number of moves to prevent infinite loops
    
    for _ in range(max_moves):
        if game.check_winner() != EMPTY:
            break
        
        board_state = get_board_state(game)
        output = net.activate(board_state)
        move = interpret_output(output, game)
        
        valid_moves = get_possible_moves(game)
        if move not in valid_moves:
            move = random.choice(valid_moves) if valid_moves else None
        
        if move:
            make_move(game, move)
        else:
            break  # No valid moves available, end the game
        
        # Switch player for the next turn
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1
    
    return calculate_fitness(game)

def get_board_state(game):
    # Flatten the board and normalize values
    return [cell / 2 for row in game.board for cell in row]

def get_possible_moves(game):
    moves = []
    if game.current_player == PLAYER1:
        current_pieces = game.p1_pieces
    else:
        current_pieces = game.p2_pieces
    
    if current_pieces < NUM_PIECES:
        # Placement phase
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == EMPTY:
                    moves.append((r, c))
    else:
        # Movement phase
        for r0 in range(BOARD_SIZE):
            for c0 in range(BOARD_SIZE):
                if game.board[r0][c0] == game.current_player:
                    for r1 in range(BOARD_SIZE):
                        for c1 in range(BOARD_SIZE):
                            if game.board[r1][c1] == EMPTY:
                                moves.append((r0, c0, r1, c1))
    return moves

def interpret_output(output, game):
    if (game.current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or \
       (game.current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
        # Placement phase
        move_index = output.index(max(output[:64]))
        return (move_index // BOARD_SIZE, move_index % BOARD_SIZE)
    else:
        # Movement phase
        from_index = output.index(max(output[:64]))
        to_index = 64 + output[64:].index(max(output[64:]))
        
        from_row, from_col = from_index // BOARD_SIZE, from_index % BOARD_SIZE
        to_row, to_col = (to_index - 64) // BOARD_SIZE, (to_index - 64) % BOARD_SIZE
        
        return (from_row, from_col, to_row, to_col)
def make_move(game, move):
    if len(move) == 2:
        if game.is_valid_placement(move[0], move[1]):
            game.place_checker(move[0], move[1])
            return True
    elif len(move) == 4:
        if game.is_valid_move(move[0], move[1], move[2], move[3]):
            game.move_checker(move[0], move[1], move[2], move[3])
            return True
    return False

def get_random_move(game):
    if (game.current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or \
       (game.current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
        # Placement phase
        empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == EMPTY]
        return random.choice(empty_cells)
    else:
        # Movement phase
        player_pieces = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == game.current_player]
        empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == EMPTY]
        from_piece = random.choice(player_pieces)
        to_cell = random.choice(empty_cells)
        return from_piece + to_cell
    
def calculate_fitness(game):
    winner = game.check_winner()
    if winner == PLAYER1:
        return 1
    elif winner == PLAYER2:
        return -1
    else:
        # If no winner, calculate based on piece difference and board control
        piece_diff = game.p1_pieces - game.p2_pieces
        control_score = sum(sum(row) for row in game.board)  # Positive for PLAYER1 control, negative for PLAYER2
        return (piece_diff + control_score * 0.1) / 10  # Normalize the score

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 50)  # Run for 50 generations
    
    # Save the winner
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run_neat(config_path)