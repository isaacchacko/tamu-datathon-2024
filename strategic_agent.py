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
        save_directory = "/C:/Users/Documents/tamu-datathon-2024"
        self.genome_file = os.path.join(save_directory, 'best_genome.pkl')

    def train(self, generations=50):
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.best_genome = self.population.run(self.evaluate_genomes, generations)
        self.save_genome()

    def evaluate_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.play_game(net)

    def play_game(self, net):
        game = Game()
        max_moves = 100
        moves = 0
        while moves < max_moves and not self.is_game_over(game):
            board_state = self.get_board_state(game)
            output = net.activate(board_state)
            move = self.interpret_output(output, game)
            game.make_move(move)
            moves += 1
        return self.calculate_fitness(game, moves)

    def is_game_over(self, game):
        return game.p1_pieces == 0 or game.p2_pieces == 0 or game.turn_count >= 32

    def calculate_fitness(self, game, moves):
        score = game.p1_pieces - game.p2_pieces
        return score + (1 / (moves + 1)) + (10 * self.evaluate_board(game))

    def get_board_state(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
        return [1 if item == current_player else (-1 if item == opponent else 0) for sublist in game.board for item in sublist]

    def interpret_output(self, output, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        if (current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or (current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
            move_index = output.index(max(output))
            return (move_index // BOARD_SIZE, move_index % BOARD_SIZE)
        else:
            from_index = output.index(max(output[:BOARD_SIZE**2]))
            to_index = BOARD_SIZE**2 + output[BOARD_SIZE**2:].index(max(output[BOARD_SIZE**2:]))
            from_row, from_col = from_index // BOARD_SIZE, from_index % BOARD_SIZE
            to_row, to_col = (to_index - BOARD_SIZE**2) // BOARD_SIZE, (to_index - BOARD_SIZE**2) % BOARD_SIZE
            if game.board[from_row][from_col] != current_player:
                return self.get_strategic_move(game)
            return (from_row, from_col, to_row, to_col)

    def get_best_move(self, game):
        if self.best_genome is None:
            return self.get_random_move(game)

        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        board_state = self.get_board_state(game)
        output = net.activate(board_state)

        possible_moves = self.get_possible_moves(game)
        move_scores = []

        for move in possible_moves:
            if len(move) == 2:  # Placement move
                move_index = move[0] * BOARD_SIZE + move[1]
                score = output[move_index]
            else:  # Movement move
                from_index = move[0] * BOARD_SIZE + move[1]
                to_index = move[2] * BOARD_SIZE + move[3]
                score = output[from_index] + output[to_index + BOARD_SIZE**2]
            move_scores.append((move, score))

        # Sort moves by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Select the best legal move
        for move, score in move_scores:
            if self.is_valid_move(game, move):
                return move

        # If no valid move found, return a random move
        return self.get_random_move(game)

    def get_strategic_move(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
        if (current_player == PLAYER1 and game.p1_pieces < NUM_PIECES) or (current_player == PLAYER2 and game.p2_pieces < NUM_PIECES):
            return self.get_best_placement(game, current_player, opponent)
        else:
            return self.get_best_movement(game, current_player, opponent)

    def get_best_placement(self, game):
        empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == EMPTY]
        if not empty_spots:
            return None  # No valid moves
        
        # Prioritize center and adjacent spots
        center = BOARD_SIZE // 2
        preferred_spots = [(r, c) for r, c in empty_spots if abs(r - center) <= 1 and abs(c - center) <= 1]
        
        if preferred_spots:
            return random.choice(preferred_spots)
        return random.choice(empty_spots)
    
    def get_best_movement(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        player_pieces = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == current_player]
        empty_spots = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] == EMPTY]
        
        if not player_pieces or not empty_spots:
            return None  # No valid moves
        
        best_move = None
        best_score = float('-inf')
        
        for from_pos in player_pieces:
            for to_pos in empty_spots:
                score = self.evaluate_move(game, from_pos, to_pos, current_player)
                if score > best_score:
                    best_score = score
                    best_move = (*from_pos, *to_pos)
        
        return best_move
    
    def evaluate_move(self, game, from_pos, to_pos, player):
        score = 0
        temp_game = game.copy()
        temp_game.board[from_pos[0]][from_pos[1]] = EMPTY
        temp_game.board[to_pos[0]][to_pos[1]] = player
        
        # Check if this move creates a line of three
        if self.creates_line_of_three(temp_game, to_pos, player):
            score += 100
        
        # Check if this move blocks opponent's line of three
        opponent = PLAYER2 if player == PLAYER1 else PLAYER1
        if self.creates_line_of_three(temp_game, to_pos, opponent):
            score += 50
        
        # Prefer moves towards the center
        center = BOARD_SIZE // 2
        score += (BOARD_SIZE - (abs(to_pos[0] - center) + abs(to_pos[1] - center))) * 2
        
        return score

    def creates_line_of_three(self, game, pos, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 3):
                r, c = _torus((pos[0] + i*dr, pos[1] + i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            for i in range(1, 3):
                r, c = _torus((pos[0] - i*dr, pos[1] - i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            if count >= 3:
                return True
        return False
    
    def is_winning_move(self, game, pos, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 3):
                r, c = _torus((pos[0] + i*dr, pos[1] + i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            for i in range(1, 3):
                r, c = _torus((pos[0] - i*dr, pos[1] - i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            if count >= 3:
                return True
        return False

    def is_blocking_move(self, game, from_pos, to_pos, opponent):
        temp_game = game.copy()
        temp_game.board[from_pos[0]][from_pos[1]] = EMPTY
        return self.is_winning_move(temp_game, to_pos, opponent)

    def get_push_moves(self, game, from_pos):
        push_moves = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = _torus((from_pos[0] + dr, from_pos[1] + dc))
            if game.board[r][c] != EMPTY:
                push_r, push_c = _torus((r + dr, c + dc))
                if game.board[push_r][push_c] == EMPTY:
                    push_moves.append((push_r, push_c))
        return push_moves

    def is_beneficial_push(self, game, from_pos, to_pos, player, opponent):
        temp_game = game.copy()
        temp_game.board[from_pos[0]][from_pos[1]] = EMPTY
        temp_game.board[to_pos[0]][to_pos[1]] = player
        return self.evaluate_board(temp_game) > self.evaluate_board(game)

    def creates_two_in_a_row(self, game, pos, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 2):
                r, c = _torus((pos[0] + i*dr, pos[1] + i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            for i in range(1, 2):
                r, c = _torus((pos[0] - i*dr, pos[1] - i*dc))
                if game.board[r][c] != player:
                    break
                count += 1
            if count == 2:
                return True
        return False

    def evaluate_board(self, game):
        current_player = PLAYER1 if game.current_player == 1 else PLAYER2
        opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
        player_pieces = sum(row.count(current_player) for row in game.board)
        opponent_pieces = sum(row.count(opponent) for row in game.board)
        player_clusters = self.count_clusters(game, current_player)
        opponent_clusters = self.count_clusters(game, opponent)
        center_control = sum(game.board[r][c] == current_player for r in range(1, BOARD_SIZE-1) for c in range(1, BOARD_SIZE-1))
        return (player_pieces - opponent_pieces) * 2 + (player_clusters - opponent_clusters) * 3 + center_control

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

    def is_better_move(self, game, move1, move2):
        game_copy1 = game.copy()
        game_copy1.make_move(move1)
        score1 = self.evaluate_board(game_copy1)

        game_copy2 = game.copy()
        game_copy2.make_move(move2)
        score2 = self.evaluate_board(game_copy2)

        return score1 > score2

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
