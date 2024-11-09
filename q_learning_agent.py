import numpy as np
import tensorflow as tf
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus

class QLearningAgent:
    def __init__(self, player, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(BOARD_SIZE * BOARD_SIZE,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(BOARD_SIZE * BOARD_SIZE)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def get_state(self, game):
        return game.board.flatten()
    
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
        state = self.get_state(game)
        possible_moves = self.get_possible_moves(game)
        
        if np.random.rand() < self.exploration_rate:
            return random.choice(possible_moves)
        
        q_values = self.model.predict(state.reshape(1, -1))[0]
        best_move = None
        best_q_value = float('-inf')
        
        for move in possible_moves:
            if len(move) == 2:
                move_index = move[0] * BOARD_SIZE + move[1]
            else:
                move_index = move[2] * BOARD_SIZE + move[3]
            
            if q_values[move_index] > best_q_value:
                best_q_value = q_values[move_index]
                best_move = move
        
        return best_move
    
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(next_state.reshape(1, -1))[0]
            target += self.discount_factor * np.max(next_q_values)
        
        current_q_values = self.model.predict(state.reshape(1, -1))[0]
        
        if len(action) == 2:
            action_index = action[0] * BOARD_SIZE + action[1]
        else:
            action_index = action[2] * BOARD_SIZE + action[3]
        
        current_q_values[action_index] = target
        
        self.model.fit(state.reshape(1, -1), current_q_values.reshape(1, -1), verbose=0)
        
        if done:
            self.exploration_rate *= self.exploration_decay

    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)