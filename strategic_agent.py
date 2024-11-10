import neat
import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES

class NEATAgent:
    def __init__(self, config_file):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        self.population = neat.Population(self.config)
        self.best_genome = None

    def train(self, generations=10000):
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

        self.best_genome = self.population.run(self.evaluate_genomes, generations)

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.play_game(net)

    def play_game(self, net):
        game = Game()
        while not game.is_terminal():
            board_state = self.get_board_state(game)
            output = net.activate(board_state)
            move = self.interpret_output(output, game)
            game.make_move(move)
        return game.p1_pieces - game.p2_pieces  # Simple fitness function

    def get_board_state(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
        return [1 if item == current_player else (-1 if item == opponent else 0) for sublist in game.board for item in sublist]

    def interpret_output(self, output, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        if (current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or (current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
            # Placement phase
            move_index = output.index(max(output))
            return (move_index // BOARD_SIZE, move_index % BOARD_SIZE)
        else:
            # Movement phase
            from_index = output.index(max(output[:BOARD_SIZE**2]))
            to_index = BOARD_SIZE**2 + output[BOARD_SIZE**2:].index(max(output[BOARD_SIZE**2:]))
            
            from_row, from_col = from_index // BOARD_SIZE, from_index % BOARD_SIZE
            to_row, to_col = (to_index - BOARD_SIZE**2) // BOARD_SIZE, (to_index - BOARD_SIZE**2) % BOARD_SIZE
            
            # Ensure the 'from' position contains the player's piece
            if game.board[from_row][from_col] != current_player:
                # If not, choose a random valid move
                return self.get_random_move(game)
            
            return (from_row, from_col, to_row, to_col)

    def get_best_move(self, game):
        if self.best_genome is None:
            return self.get_random_move(game)
        
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        board_state = self.get_board_state(game)
        output = net.activate(board_state)
        return self.interpret_output(output, game)

    def get_random_move(self, game):
        possible_moves = self.get_possible_moves(game)
        return random.choice(possible_moves)

    def get_possible_moves(self, game):
        moves = []
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        if game.p1_pieces < NUM_PIECES or game.p2_pieces < NUM_PIECES:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves