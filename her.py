from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import cases
from cases import get_best_move_and_fitness, run_cases

class HisAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.lastFitness = 0

    # given the game state, gets all of the possible moves
    def get_best_move(self, game, board, turn_count, attempt_number):
        move, self.lastFitness = get_best_move_and_fitness(run_cases(game, board, turn_count, attempt_number, self.lastFitness))
