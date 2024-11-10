import random
import math
import copy
from PushBattle import PLAYER1, PLAYER2, EMPTY, BOARD_SIZE  # Import constants used in this file

class TreeNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = self._generate_possible_moves()

    def _apply_move(self, game_state, move):
        """Apply a move to a game state, generating a new game state."""
        next_state = copy.deepcopy(game_state)

        if len(move) == 2:  # Placement move
            row, col = move
            next_state.board[row][col] = game_state.current_player
        elif len(move) == 4:  # Movement move
            start_row, start_col, end_row, end_col = move
            next_state.board[end_row][end_col] = next_state.board[start_row][start_col]
            next_state.board[start_row][start_col] = EMPTY

        next_state.current_player *= -1
        return next_state

    def _generate_possible_moves(self):
        """Generates possible moves based on game state."""
        board = self.game_state.board
        moves = []

        current_player_pieces = self.game_state.p1_pieces if self.game_state.current_player == PLAYER1 else self.game_state.p2_pieces
        if current_player_pieces < 8:
            for row in range(len(board)):
                for col in range(len(board[row])):
                    if board[row][col] == EMPTY:
                        moves.append((row, col))
        else:
            for start_row in range(len(board)):
                for start_col in range(len(board[start_row])):
                    if board[start_row][start_col] == self.game_state.current_player:
                        for end_row, end_col in get_adjacent_positions(start_row, start_col, board):
                            if board[end_row][end_col] == EMPTY:
                                moves.append((start_row, start_col, end_row, end_col))

        return moves

    def is_terminal(self):
        """Determine if the game state is terminal."""
        return self.game_state.check_winner() is not None or self.game_state.is_draw()

    def is_fully_expanded(self):
        """Check if all possible moves from this node's game state have been expanded."""
        return len(self.children) == len(self.untried_moves)

    def expand(self):
        """Expand a node by adding a new child for an untried move."""
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop()
        next_state = self._apply_move(self.game_state, move)
        child_node = TreeNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=0.8):
        """Select the child with the highest UCB1 score."""
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) +
                              exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        )

    def simulate(self, game_state, player, max_depth=10):
        """Simulate a playthrough up to max_depth using a heuristic-driven move selection."""
        depth = 0
        while depth < max_depth and not self.is_terminal():
            possible_moves = generate_possible_moves(game_state)
            if not possible_moves:
                break
            move = heuristic_choice(possible_moves, game_state)
            game_state = self._apply_move(game_state, move)
            depth += 1

        winner = game_state.check_winner()
        return 1 if winner == player else -1 if winner is not None else 0

    def backpropagate(self, result):
        """Backpropagate the result of a simulation up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            result = -result
            node = node.parent

def get_adjacent_positions(row, col, board):
    """Get adjacent positions for movement on the board."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    adjacent_positions = []

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):
            adjacent_positions.append((new_row, new_col))

    return adjacent_positions

def generate_possible_moves(game_state):
    """Generate all possible moves for the current game state."""
    possible_moves = []
    board = game_state.board

    if (game_state.current_player == PLAYER1 and game_state.p1_pieces < 8) or \
       (game_state.current_player == PLAYER2 and game_state.p2_pieces < 8):
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == EMPTY:
                    possible_moves.append((row, col))
    else:
        for start_row in range(len(board)):
            for start_col in range(len(board[0])):
                if board[start_row][start_col] == game_state.current_player:
                    for end_row, end_col in get_adjacent_positions(start_row, start_col, board):
                        if board[end_row][end_col] == EMPTY:
                            possible_moves.append((start_row, start_col, end_row, end_col))

    return possible_moves

def heuristic_choice(moves, game_state):
    """Prioritize moves closer to the center for better positioning."""
    center = BOARD_SIZE // 2
    return min(moves, key=lambda move: abs(move[0] - center) + abs(move[1] - center))

class IBAgent:
    def __init__(self, player, simulations=300):
        self.player = player
        self.simulations = simulations

    def get_best_move(self, game_state):
        """Run MCTS and return the best move found."""
        root = TreeNode(game_state, parent=None)

        for _ in range(self.simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            if not node.is_terminal():
                node = node.expand() or node

            result = node.simulate(node.game_state, self.player)
            node.backpropagate(result)

        if root.children:
            best_move_node = max(root.children, key=lambda child: child.wins / child.visits)
            best_move = best_move_node.move
        else:
            possible_moves = generate_possible_moves(game_state)
            best_move = random.choice(possible_moves) if possible_moves else None

        if best_move:
            if isinstance(best_move, tuple) and len(best_move) == 4:
                start_pos, end_pos = best_move[:2], best_move[2:]
                return (start_pos[0], start_pos[1], end_pos[0], end_pos[1])
            else:
                return best_move
        return None