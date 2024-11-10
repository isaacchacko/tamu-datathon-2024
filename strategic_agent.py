import neat
import numpy as np
import random
import pickle
import os

BOARD_SIZE = 8
NUM_PIECES = 8

class NEATAgent:
    def __init__(self, config_file):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_file)
        self.population = neat.Population(self.config)
        self.best_genome = None
        self.genome_file = 'best_genome.pkl'

    def train(self, generations=50):
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.best_genome = self.population.run(self.evaluate_genomes, generations)
        self.save_genome()

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.evaluate_fitness(net)

    def evaluate_fitness(self, net):
        total_score = 0
        for _ in range(5):  # Play 5 games and average the score
            game = PushBattleGame()
            score = self.play_game(net, game)
            total_score += score
        return total_score / 5

    def play_game(self, net, game):
        while not game.is_game_over():
            move = self.get_move(net, game)
            game.make_move(move)
            if game.check_win():
                return 1 if game.current_player == 1 else -1
        return 0  # Draw

    def get_move(self, net, game):
        board_state = self.get_board_state(game)
        output = net.activate(board_state)
        return self.interpret_output(output, game)

    def get_board_state(self, game):
        state = []
        for row in game.board:
            for cell in row:
                if cell == 1:
                    state.append(1)
                elif cell == 2:
                    state.append(-1)
                else:
                    state.append(0)
        state.append(1 if game.current_player == 1 else -1)
        state.append(game.pieces_placed[0] / NUM_PIECES)
        state.append(game.pieces_placed[1] / NUM_PIECES)
        return state

    def interpret_output(self, output, game):
        if game.pieces_placed[game.current_player - 1] < NUM_PIECES:
            # Placement move
            move_index = np.argmax(output[:BOARD_SIZE**2])
            return (move_index // BOARD_SIZE, move_index % BOARD_SIZE)
        else:
            # Movement move
            from_index = np.argmax(output[:BOARD_SIZE**2])
            to_index = np.argmax(output[BOARD_SIZE**2:]) + BOARD_SIZE**2
            from_pos = (from_index // BOARD_SIZE, from_index % BOARD_SIZE)
            to_pos = ((to_index - BOARD_SIZE**2) // BOARD_SIZE, (to_index - BOARD_SIZE**2) % BOARD_SIZE)
            return (*from_pos, *to_pos)

    def save_genome(self):
        with open(self.genome_file, 'wb') as f:
            pickle.dump(self.best_genome, f)
        print(f"Best genome saved to {self.genome_file}")

    def load_genome(self):
        if os.path.exists(self.genome_file):
            with open(self.genome_file, 'rb') as f:
                self.best_genome = pickle.load(f)
            print(f"Loaded existing genome from {self.genome_file}")
            return True
        return False

    def get_best_move(self, game):
        if self.best_genome is None:
            return self.get_random_move(game)
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        return self.get_move(net, game)

    def get_random_move(self, game):
        if game.pieces_placed[game.current_player - 1] < NUM_PIECES:
            # Placement move
            empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == 0]
            return random.choice(empty_spots)
        else:
            # Movement move
            player_pieces = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == game.current_player]
            empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == 0]
            from_pos = random.choice(player_pieces)
            to_pos = random.choice(empty_spots)
            return (*from_pos, *to_pos)

class PushBattleGame:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1
        self.pieces_placed = [0, 0]

    def make_move(self, move):
        if len(move) == 2:
            # Placement move
            r, c = move
            if self.board[r][c] == 0:
                self.board[r][c] = self.current_player
                self.pieces_placed[self.current_player - 1] += 1
                self.push_adjacent(r, c)
                self.current_player = 3 - self.current_player  # Switch player
                return True
        elif len(move) == 4:
            # Movement move
            r0, c0, r1, c1 = move
            if self.board[r0][c0] == self.current_player and self.board[r1][c1] == 0:
                self.board[r0][c0] = 0
                self.board[r1][c1] = self.current_player
                self.push_adjacent(r1, c1)
                self.current_player = 3 - self.current_player  # Switch player
                return True
        return False

    def push_adjacent(self, r, c):
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            self.push_piece(r + dr, c + dc, dr, dc)

    def push_piece(self, r, c, dr, dc):
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] != 0:
            next_r, next_c = (r + dr) % BOARD_SIZE, (c + dc) % BOARD_SIZE
            if self.board[next_r][next_c] == 0:
                self.board[next_r][next_c] = self.board[r][c]
                self.board[r][c] = 0
            else:
                self.push_piece(next_r, next_c, dr, dc)

    def check_win(self):
        for player in [1, 2]:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self.check_three_in_a_row(r, c, player):
                        return True
        return False

    def check_three_in_a_row(self, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 0
            for i in range(3):
                nr, nc = (r + i*dr) % BOARD_SIZE, (c + i*dc) % BOARD_SIZE
                if self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            if count == 3:
                return True
        return False

    def is_game_over(self):
        return self.check_win() or all(pieces == NUM_PIECES for pieces in self.pieces_placed)