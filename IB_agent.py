import random
import math
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class TreeNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state  # Current state of the game at this node
        self.parent = parent  # Parent node
        self.children = {}  # Dictionary of children {move: TreeNode}
        self.visits = 0  # Number of times this node was visited
        self.wins = 0  # Number of wins for the current player at this node

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_state.get_possible_moves())

    def best_child(self, exploration_weight=1.4):
        """Selects the best child based on the UCB1 formula."""
        return max(self.children.values(), key=lambda child: child.wins / child.visits + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits))

    def expand(self):
        """Expands the node by creating a child for an untried move."""
        untried_moves = [move for move in self.game_state.get_possible_moves() if move not in self.children]
        move = random.choice(untried_moves)
        next_state = self.game_state.make_move(move)
        child_node = TreeNode(next_state, parent=self)
        self.children[move] = child_node
        return child_node

    def update(self, result):
        """Updates this node with the result of a simulated game."""
        self.visits += 1
        if result == self.game_state.current_player:
            self.wins += 1
class IBAgent:
    def __init__(self, player=PLAYER2, simulations=100):
        self.player = player
        self.simulations = simulations  # This ensures `simulations` is initialized correctly.

    def get_possible_moves(self, game):
        """Generate possible moves based on the number of pieces on the board."""
        moves = []
        current_pieces = game.p1_pieces if self.player == PLAYER1 else game.p2_pieces

        if current_pieces < NUM_PIECES:
            # Placement moves: generate all empty positions
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if game.board[row][col] == EMPTY:
                        moves.append((row, col))  # Placement move format
        else:
            # Movement moves: generate moves from one cell to another empty cell
            for start_row in range(BOARD_SIZE):
                for start_col in range(BOARD_SIZE):
                    if game.board[start_row][start_col] == self.player:
                        for end_row in range(BOARD_SIZE):
                            for end_col in range(BOARD_SIZE):
                                if game.board[end_row][end_col] == EMPTY:
                                    moves.append((start_row, start_col, end_row, end_col))  # Movement move format
        return moves


    def get_best_move(self, game):
        """Uses MCTS to select the best move for the current game state.
        root = TreeNode(game)

        for _ in range(self.simulations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            
            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            result = self.simulate(node.game_state)
            
            # Backpropagation
            while node:
                node.update(result)
                node = node.parent

        if not root.children:
            print("Error: No valid moves found by MCTS.")
            return None

        # Choose the move of the child with the highest visit count
        best_move = max(root.children, key=lambda move: root.children[move].visits)
        print(f"Selected move by IBAgent: {best_move}")  # Debugging output
        return best_move"""

        """Temporarily returns a random move for debugging purposes."""
        possible_moves = self.get_possible_moves(game)
        selected_move = random.choice(possible_moves)
        print(f"Debugging - Randomly selected move by IBAgent: {selected_move}")
        return selected_move



    def simulate(self, game_state):
        """Simulates a random playout from the given game state."""
        while not game_state.is_terminal():
            possible_moves = self.get_possible_moves(game_state)
            if not possible_moves:
                print("Error: No possible moves during simulation.")
                return None  # This handles cases where no moves are possible

            move = random.choice(possible_moves)
            game_state = game_state.make_move(move)  # Ensure this applies the move correctly
            print(f"Simulated move: {move} -> New state: {game_state}")
        return game_state.get_winner()  # Assume the game state can determine a winner

