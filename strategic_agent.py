import neat
import random
import pickle
import os
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class NEATAgent:
    def __init__(self, config_file):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        self.best_genome = None
        self.genome_file = 'best_genome.pkl'
        self.load_genome()

    def load_genome(self):
        if os.path.exists(self.genome_file):
            with open(self.genome_file, 'rb') as f:
                self.best_genome = pickle.load(f)
            print(f"Best genome loaded from {self.genome_file}")
            return True
        print(f"No genome file found at {self.genome_file}")
        return False

    def get_board_state(self, game):
        return [cell / 2 for row in game.board for cell in row] + [game.current_player / 2]

    def interpret_output(self, output, game):
        if (game.current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or \
           (game.current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
            # Placement phase
            move_index = output.index(max(output))
            return (move_index // BOARD_SIZE, move_index % BOARD_SIZE)
        else:
            # Movement phase
            from_index = output.index(max(output[:BOARD_SIZE**2]))
            to_index = BOARD_SIZE**2 + output[BOARD_SIZE**2:].index(max(output[BOARD_SIZE**2:]))
            
            from_row, from_col = from_index // BOARD_SIZE, from_index % BOARD_SIZE
            to_row, to_col = (to_index - BOARD_SIZE**2) // BOARD_SIZE, (to_index - BOARD_SIZE**2) % BOARD_SIZE
            
            return (from_row, from_col, to_row, to_col)
    def get_best_move(self, game):
        if self.best_genome is None:
            return self.get_random_move(game)
        
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        board_state = self.get_board_state(game)
        output = net.activate(board_state)
        
        # Get all valid moves
        valid_moves = self.get_possible_moves(game)
        
        # Filter and sort moves based on network output
        if len(valid_moves[0]) == 2:  # Placement phase
            sorted_moves = sorted([(move, output[move[0]*BOARD_SIZE + move[1]]) for move in valid_moves], 
                                key=lambda x: x[1], reverse=True)
        else:  # Movement phase
            sorted_moves = sorted([(move, output[move[0]*BOARD_SIZE + move[1]] + output[BOARD_SIZE**2 + move[2]*BOARD_SIZE + move[3]]) 
                                for move in valid_moves], key=lambda x: x[1], reverse=True)
        
        # Return the best valid move
        return sorted_moves[0][0]

    def is_valid_move(self, game, move):
        if len(move) == 2:
            return game.is_valid_placement(move[0], move[1])
        elif len(move) == 4:
            return game.is_valid_move(move[0], move[1], move[2], move[3])
        return False

    def get_random_move(self, game):
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

    # You can remove or comment out the following methods if they're not needed:
    # train, save_genome, evaluate_genomes, play_game, evaluate_board, count_clusters, dfs, predict_opponent_move, is_better_move