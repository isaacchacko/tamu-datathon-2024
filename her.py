import numpy as np
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import cases
from cases import get_best_move_and_fitness, run_cases
from random_agent import RandomAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class HisAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.lastFitness = 0
        self.random = RandomAgent()
        self.random_move_count = 0

    def get_best_move(self, game, board, turn_count, attempt_number):
        result = get_best_move_and_fitness(
            run_cases(game, board, turn_count, attempt_number, self.lastFitness))

        if result == False:
            self.lastFitness = 1
            self.random_move_count += 1
            logging.warning(f"Running random case. Random move count: {
                            self.random_move_count}")
            return self.random.get_best_move(game)

        else:
            move, self.lastFitness = result
            if not isinstance(move, (list, tuple)) or len(move) not in [2, 4]:
                self.lastFitness = 1
                self.random_move_count += 1
                logging.warning(f"Invalid move format. Running random case. Random move count: {
                                self.random_move_count}")
                return self.random.get_best_move(game)

            # Convert NumPy int64 to Python int
            move = [int(x) for x in move]
            logging.info(f"Using calculated move: {
                         move} with fitness: {self.lastFitness}")
            return move
