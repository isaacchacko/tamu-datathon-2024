import json
import random
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus, chess_notation_to_array, array_to_chess_notation


class RandomAgent:
    def __init__(self, player=PLAYER1):
        self.player = player

    def get_possible_moves(self, game):
        moves = []
        current_pieces = game.p1_pieces if game.current_player == PLAYER1 else game.p2_pieces
        if current_pieces < NUM_PIECES:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.board[r][c] == EMPTY:
                        moves.append((r, c))
        else:
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game.board[r0][c0] == game.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game.board[r1][c1] == EMPTY:
                                    moves.append((r0, c0, r1, c1))
        return moves

    def get_best_move(self, game):
        possible_moves = self.get_possible_moves(game)
        return random.choice(possible_moves)


def simulate_game():
    game = Game()
    game_string = ""
    agent1 = RandomAgent(PLAYER1)
    agent2 = RandomAgent(PLAYER2)

    while True:
        current_agent = agent1 if game.current_player == PLAYER1 else agent2
        move = current_agent.get_best_move(game)
        chess_move = array_to_chess_notation(move)

        game_string += f"-{chess_move}"

        if len(move) == 2:
            game.place_checker(move[0], move[1])
        else:
            game.move_checker(move[0], move[1], move[2], move[3])

        winner = game.check_winner()
        if winner != EMPTY:
            return game_string, "PLAYER1" if winner == PLAYER1 else "PLAYER2"

        game.current_player *= -1


def main():
    num_simulations = int(input("Enter the number of simulations to run: "))

    results = []
    try:

        for i in range(num_simulations):
            game_string, winner = simulate_game()
            results.append({"game_string": game_string, "winner": winner})
            if i % 1000 == 999:
                print(f"Simulation {i+1}/{num_simulations} completed")

    except KeyboardInterrupt:
        pass

    finally:
        output_file = "game_simulations.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results have been written to {output_file}")


if __name__ == "__main__":
    main()
