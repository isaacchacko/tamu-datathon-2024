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
        self.population = neat.Population(self.config)
        self.best_genome = None
        self.genome_file = 'best_genome.pkl'

    def train(self, generations=50):
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

        self.best_genome = self.population.run(self.evaluate_genomes, generations)
        self.save_genome()

    def save_genome(self):
        version = len([f for f in os.listdir() if f.startswith('best_genome')]) + 1
        genome_file = f'best_genome_v{version}.pkl'
        with open(genome_file, 'wb') as f:
            pickle.dump(self.best_genome, f)
        print(f"Best genome saved to {genome_file}")


    def load_best_genome(self):
        genome_files = [f for f in os.listdir() if f.startswith('best_genome')]
        if genome_files:
            latest_genome = max(genome_files, key=os.path.getctime)  # Load most recent genome
            with open(latest_genome, 'rb') as f:
                self.best_genome = pickle.load(f)
            print(f"Best genome loaded from {latest_genome}")
            return True
        return False

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.play_game(net)
            # Additional evaluations can be added here

    def play_game(self, net):
        game = Game()
        max_moves = 100  # Set a maximum number of moves to prevent infinite loops
        moves = 0
        while moves < max_moves and not self.is_game_over(game):
            board_state = self.get_board_state(game)
            output = net.activate(board_state)
            move = self.interpret_output(output, game)
            game.make_move(move)
            moves += 1
        
        return self.calculate_fitness(game, moves)

    def is_game_over(self, game):
        # Implement game-over conditions here
        # For example:
        return game.p1_pieces == 0 or game.p2_pieces == 0 or game.turn_count >= 32

    def calculate_fitness(self, game, moves):
        """
        Calculate the fitness score based on game outcome and board control.
        """
        # Initial fitness based on win/loss/draw
        winner = game.check_winner()
        if winner == PLAYER1:
            return 1000  # Large reward for winning
        elif winner == PLAYER2:
            return -1000  # Large penalty for losing
        elif game.turn_count >= 32:
            return 10  # Small reward for draw (if applicable)

        # Fitness based on board control and move efficiency
        fitness = 0

        # Board control - center and corner control
        fitness += self.evaluate_board_control(game)

        # Piece pushing - reward pushing opponent's pieces into bad positions
        fitness += self.evaluate_piece_pushing(game)

        # Move efficiency - reward forming two-in-a-row or blocking opponent's rows
        fitness += self.evaluate_move_efficiency(game)

        # Shorter games are better, so we add a small bonus for faster wins
        fitness += 1 / moves if moves > 0 else 0

        return fitness

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
            if not self.load_genome():
                return self.get_random_move(game)
        
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        board_state = self.get_board_state(game)
        output = net.activate(board_state)
        move = self.interpret_output(output, game)
        
        # Simple opponent modeling
        opponent_move = self.predict_opponent_move(game)
        if self.is_better_move(game, move, opponent_move):
            return move
        else:
            return self.get_random_move(game)

    def get_random_move(self, game):
        possible_moves = self.get_possible_moves(game)
        return random.choice(possible_moves)

    def get_possible_moves(self, game):
        moves = []
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        if (current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or (current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
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
    
    def evaluate_board_control(self, game):
        """
        Reward controlling key areas of the board like center and corners.
        """
        center_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        corner_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
        
        fitness = 0
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == PLAYER1:
                    if (r, c) in center_positions:
                        fitness += 5  # Higher reward for controlling center
                    elif (r, c) in corner_positions:
                        fitness += 3  # Slightly lower reward for controlling corners

                elif game.board[r][c] == PLAYER2:
                    if (r, c) in center_positions:
                        fitness -= 5  # Penalize if opponent controls center
                    elif (r, c) in corner_positions:
                        fitness -= 3  # Penalize if opponent controls corners

        return fitness
    
    def evaluate_piece_pushing(self, game):
        """
        Reward pushing opponent's pieces into unfavorable positions.
        Penalize if your own pieces are pushed.
        """
        push_reward = 0
        
        # Check neighboring tiles and see if any pushes occurred
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == PLAYER1:
                    push_reward += self.check_push_effect(game, r, c)
        
        return push_reward
    
    def check_push_effect(self, game, r0, c0):
        """
        Evaluate whether pushing occurred around a given tile.
        """
        dirs = [(-1,-1), (-1,0), (-1,+1), (0,+1), (+1,+1), (+1,0), (+1,-1), (0,-1)]
        
        push_score = 0
        
        for dr, dc in dirs:
            r1, c1 = _torus(r0 + dr, c0 + dc)
            if game.board[r1][c1] == PLAYER2:
                r2, c2 = _torus(r1 + dr, c1 + dc)
                if game.board[r2][c2] == EMPTY:
                    push_score += 10   # Reward pushing an opponent piece into an empty spot
                    
            elif game.board[r1][c1] == PLAYER1:
                r2, c2 = _torus(r1 + dr, c1 + dc)
                if game.board[r2][c2] != EMPTY:
                    push_score -= 5    # Penalize if our own piece is pushed

        return push_score
    
    def evaluate_move_efficiency(self, game):
        """
        Reward efficient moves that form two-in-a-row or block opponent's rows.
        Penalize inefficient/random moves.
        """
        efficiency_score = 0
        
        # Check rows/columns/diagonals for two-in-a-row formations
        efficiency_score += self.count_two_in_a_row(game.board, PLAYER1) * 15
        efficiency_score -= self.count_two_in_a_row(game.board, PLAYER2) * 15
        
        return efficiency_score
    
    def count_two_in_a_row(self, board, player):
        """
        Count occurrences of two aligned pieces on rows/columns/diagonals.
        """
        
        count = 0
        
        # Check rows and columns
        for i in range(BOARD_SIZE):
            count += self.check_line(board[i], player)   # Rows
            count += self.check_line([board[j][i] for j in range(BOARD_SIZE)], player)   # Columns
        
        # Check diagonals
        count += self.check_diagonal(board, player)
        
        return count
    
    def check_line(self, line, player):
        """
       Check horizontal/vertical lines for two-in-a-row patterns.
       """
        cnt = sum(1 for i in range(len(line)-2) if line[i:i+2].count(player) == 2)
        return cnt
    
    def check_diagonal(self, board, player):
        """
        Check both main diagonals and anti-diagonals for two-in-a-row patterns.
        """
        count = 0

        # Check main diagonals (top-left to bottom-right)
        for i in range(BOARD_SIZE - 2):  # We check up to BOARD_SIZE - 2 because we are looking for 2-in-a-row
            if board[i][i] == player and board[i + 1][i + 1] == player:
                count += 1

        # Check anti-diagonals (top-right to bottom-left)
        for i in range(BOARD_SIZE - 2):
            if board[i][BOARD_SIZE - 1 - i] == player and board[i + 1][BOARD_SIZE - 2 - i] == player:
                count += 1

        return count
    
    def evaluate_board(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
        
        player_pieces = sum(row.count(current_player) for row in game.board)
        opponent_pieces = sum(row.count(opponent) for row in game.board)
        
        player_clusters = self.count_clusters(game, current_player)
        opponent_clusters = self.count_clusters(game, opponent)
        
        return (player_pieces - opponent_pieces) + (player_clusters - opponent_clusters)

    def count_clusters(self, game, player):
        visited = set()
        clusters = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == player and (r, c) not in visited:
                    self.dfs(game, r, c, player, visited)
                    clusters += 1
        return clusters

    def dfs(self, game, r, c, player, visited):
        if (r, c) in visited or game.board[r][c] != player:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = _torus((r + dr, c + dc))
            self.dfs(game, nr, nc, player, visited)
            
    def predict_opponent_move(self, game):
        # Implement a simple prediction of the opponent's next move
        # This could be based on their previous moves or a heuristic
        pass

    def is_better_move(self, game, our_move, opponent_move):
        # Implement logic to compare our move with the predicted opponent move
        # Return True if our move is better, False otherwise
        pass
    
    def get_best_move(self, game):
        if self.best_genome is None:
            return self.get_random_move(game)
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        board_state = self.get_board_state(game)
        output = net.activate(board_state)
        move = self.interpret_output(output, game)
                # Simple opponent modeling
        opponent_move = self.predict_opponent_move(game)
        if self.is_better_move(game, move, opponent_move):
            return move
        else:
            return self.get_random_move(game)
        
'''


'''