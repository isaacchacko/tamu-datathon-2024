import random
import math
import copy

class TreeNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = self._generate_possible_moves()  # Initialize possible moves

    def _apply_move(self, game_state, move):
        """Apply a move to a game state, generating a new game state."""
        # Create a deep copy of game_state if necessary
        next_state = copy.deepcopy(game_state)

        if len(move) == 2:  # Placement move
            row, col = move
            next_state.board[row][col] = game_state.current_player
        elif len(move) == 4:  # Movement move
            start_row, start_col, end_row, end_col = move
            next_state.board[end_row][end_col] = next_state.board[start_row][start_col]
            next_state.board[start_row][start_col] = 0  # Clear original spot


        # Update any additional state variables if needed
        next_state.current_player *= -1  # Switch players
        return next_state
    
    def _pieces_left_to_place(self):
        """Determine if pieces are still being placed based on game state."""
        return self.game_state.p1_pieces < 8 or self.game_state.p2_pieces < 8  # Adjust based on rules
    
    def _adjacent_positions(self, row, col):
        """Get valid adjacent positions for movement on an 8x8 board."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        adjacent = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:  # Stay within bounds
                adjacent.append((new_row, new_col))
        return adjacent
    
    def _generate_possible_moves(self):
        # Check if pieces are still being placed or moved
        board = self.game_state.board
        moves = []
        
        # Generate placement moves (two integers)
        if self._pieces_left_to_place():
            for row in range(len(board)):
                for col in range(len(board[row])):
                    if board[row][col] == 0:  # Assume 0 represents an empty space
                        moves.append((row, col))
        else:
            # Generate movement moves (four integers)
            for start_row in range(len(board)):
                for start_col in range(len(board[start_row])):
                    if board[start_row][start_col] == self.game_state.current_player:  # Current player piece
                        for end_row, end_col in self._adjacent_positions(start_row, start_col):
                            if board[end_row][end_col] == 0:  # Check if end position is empty
                                moves.append((start_row, start_col, end_row, end_col))
        return generate_possible_moves(self.game_state)
    
    def _is_valid_move(self, row, col):
        """Determines if placing a piece at (row, col) is valid based on current game state."""
        # Check if the cell is empty; adjust this condition if necessary
        return self.game_state.board[row][col] == 0  # Assuming 0 means an empty cell
    
    def make_move(self, move):
        """Simulates making a move by creating a new game state based on a copy of the current state."""
        new_state = copy.deepcopy(self.game_state)  # Create a deep copy of the current game state
        row, col = move
        # Assuming current_player is represented by -1 or 1
        new_state.board[row][col] = new_state.current_player
        # Update game logic for turn management, pieces count, etc., if necessary
        new_state.current_player *= -1  # Switch player turn
        return new_state
    
    def is_terminal(self):
        """Determine if the game state is terminal (game over)."""
        # Check if there is a winner or if the game has reached a final state.
        # This is an assumed method based on a typical game API. Adapt as needed.
        return self.game_state.check_winner() is not None or self.game_state.is_draw()

    def is_fully_expanded(self):
        """Check if all possible moves from this node's game state have been expanded."""
        return len(self.children) == len(self._generate_possible_moves())

    def expand(self):
        """Expand a node by adding a new child for an untried move."""
        if not self.untried_moves:
            return None  # No moves to expand

        move = self.untried_moves.pop()
        print(f"Expanding move: {move}")

        next_state = self._apply_move(self.game_state, move)
        child_node = TreeNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=1.4):
        return max(
            self.children.values(),  # Use `.values()` to get child nodes
            key=lambda child: (child.wins / child.visits) +
                              exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        )


    def simulate(self, game_state, player):
        while not self.is_terminal():  # No arguments needed
            possible_moves = game_state.get_possible_moves()
            if not possible_moves:
                break  # No moves left; exit simulation
            
            move = random.choice(possible_moves)
            game_state = self._apply_move(game_state, move)

        winner = game_state.check_winner()
        return 1 if winner == player else -1 if winner is not None else 0

    def backpropagate(self, result):
        node = self
        while node is not None:
            # Check if node is an instance of TreeNode; otherwise, raise error with context.
            if not isinstance(node, TreeNode):
                raise TypeError(
                    f"Backpropagation encountered a non-TreeNode instance: {type(node)}. "
                    f"Node details: {node}, current TreeNode: {self}"
                )
            
            node.visits += 1  # Increment visit count
            node.wins += result  # Update wins with result
            result = -result  # Flip result for the opponent
            
            # Move up to the parent
            node = node.parent

# Helper function to get adjacent positions
def get_adjacent_positions(row, col, board):
    adjacent_positions = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):  # Ensure within bounds
            adjacent_positions.append((new_row, new_col))

    return adjacent_positions

def generate_possible_moves(game_state):
        possible_moves = []
        board = game_state.board  # Assuming game_state has a board attribute

        # Check if all pieces are already on the board or more can be placed
        if game_state.turn_count < 16:  # Assuming 16 turns for initial placements
            for row in range(len(board)):
                for col in range(len(board[0])):
                    if board[row][col] == 0:  # Assuming empty spaces are marked with 0
                        possible_moves.append((row, col))  # Two-integer format for placement moves
        else:
            # Generate four-integer moves if all pieces are already placed
            for start_row in range(len(board)):
                for start_col in range(len(board[0])):
                    if board[start_row][start_col] == game_state.current_player:
                        # Check for possible moves to adjacent cells
                        for end_row, end_col in get_adjacent_positions(start_row, start_col, board):
                            if board[end_row][end_col] == 0:  # Empty destination
                                possible_moves.append(((start_row, start_col), (end_row, end_col)))

        return possible_moves

class IBAgent:
    def __init__(self, player, simulations=100):
        self.player = player
        self.simulations = simulations

    def get_adjacent_positions(row, col, board):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        adjacent_positions = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < len(board) and 0 <= new_col < len(board[0]):
                adjacent_positions.append((new_row, new_col))
        
        return adjacent_positions

    def get_best_move(self, game_state):
        root = TreeNode(game_state, parent=None)

        # Run simulations
        for _ in range(self.simulations):
            node = root
            # Selection and Expansion
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            if not node.is_terminal():
                node = node.expand() or node  # Expand if possible

            # Simulation and Backpropagation
            result = node.simulate(node.game_state, self.player)
            node.backpropagate(result)

        # Choose the best move
        if root.children:
            best_move_node = max(root.children, key=lambda child: child.wins / child.visits)
            best_move = best_move_node.move
        else:
            # Fallback if no children were expanded
            possible_moves = generate_possible_moves(game_state)
            best_move = random.choice(possible_moves) if possible_moves else None

        # Format the move for the game
        if best_move:
            print(f'{best_move = }')
            if isinstance(best_move[0], tuple):  # Four-integer move format
                start_pos, end_pos = best_move
                print(start_pos, end_pos)
                return (start_pos[0], start_pos[1], end_pos[0], end_pos[1])
            else:  # Two-integer placement format
                return (best_move[0], best_move[1])
        return None