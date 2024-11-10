import numpy as np
import tensorflow as tf
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES
import random

def get_available_device():
    return tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0')

class QLearningAgent:
    def __init__(self, player, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        with get_available_device():
            self.model = self._build_model()
            self.mse_loss = tf.keras.losses.MeanSquaredError()
    
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
        
        with get_available_device():
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
    
    @tf.function
    def _train_step(self, state, target):
        with tf.GradientTape() as tape:
            predictions = self.model(state)
            loss = self.mse_loss(target, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, state, action, reward, next_state, done):
        with get_available_device():
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
            
            loss = self._train_step(tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32),
                                    tf.convert_to_tensor(current_q_values.reshape(1, -1), dtype=tf.float32))
        
        if done:
            self.exploration_rate *= self.exploration_decay
        
        return loss.numpy()  # Return the loss value

    def save_model(self, filename):
        with get_available_device():
            self.model.save(filename)
    
    def load_model(self, filename):
        with get_available_device():
            self.model = tf.keras.models.load_model(filename)