import neat
import os
import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES
import pickle

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        
        for _ in range(5):
            genome.fitness += play_game(net)

def play_game(net):
    game = Game()
    max_moves = 100
    
    for _ in range(max_moves):
        if game.check_winner() != EMPTY:
            break
        
        board_state = get_board_state(game)
        output = net.activate(board_state)
        move = get_valid_move(output, game)
        
        if move:
            make_move(game, move)
        else:
            # No valid move available, end the game
            break
        
        game.current_player = PLAYER2 if game.current_player == PLAYER1 else PLAYER1
    
    return calculate_fitness(game)

def get_board_state(game):
    return [cell / 2 for row in game.board for cell in row] + [game.current_player / 2]

def get_possible_moves(game):
    moves = []
    if (game.current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or \
       (game.current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
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

def get_valid_move(output, game):
    valid_moves = get_possible_moves(game)
    if not valid_moves:
        return None

    if len(valid_moves[0]) == 2:  # Placement phase
        move_scores = [(move, output[move[0]*BOARD_SIZE + move[1]]) for move in valid_moves]
    else:  # Movement phase
        move_scores = [(move, output[move[0]*BOARD_SIZE + move[1]] + output[BOARD_SIZE**2 + move[2]*BOARD_SIZE + move[3]]) 
                       for move in valid_moves]

    return max(move_scores, key=lambda x: x[1])[0]

def make_move(game, move):
    if len(move) == 2:
        game.place_checker(move[0], move[1])
    elif len(move) == 4:
        game.move_checker(move[0], move[1], move[2], move[3])

def calculate_fitness(game):
    winner = game.check_winner()
    if winner == PLAYER1:
        return 1
    elif winner == PLAYER2:
        return -1
    else:
        piece_diff = game.p1_pieces - game.p2_pieces
        control_score = sum(sum(row) for row in game.board)
        return (piece_diff + control_score * 0.1) / 10

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 5)
    
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    run_neat(config_path)