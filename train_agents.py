from q_learning_agent import QLearningAgent, get_available_device
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY
import tensorflow as tf
import numpy as np

def play_game(agent1, agent2):
    game = Game()
    done = False
    total_loss = 0
    moves = 0
    
    while not done:
        current_agent = agent1 if game.current_player == PLAYER1 else agent2
        state = current_agent.get_state(game)
        
        move = current_agent.get_best_move(game)
        
        if len(move) == 2:
            game.place_checker(*move)
        else:
            game.move_checker(*move)
        
        next_state = current_agent.get_state(game)
        winner = game.check_winner()
        
        if winner != EMPTY:
            done = True
            reward = 1 if winner == current_agent.player else -1
        else:
            reward = 0
        
        loss = current_agent.train(state, move, reward, next_state, done)
        total_loss += loss
        moves += 1
        
        game.current_player *= -1
    
    return winner, total_loss / moves

def train_agents(num_episodes=10000):
    with get_available_device():
        agent1 = QLearningAgent(PLAYER1)
        agent2 = QLearningAgent(PLAYER2)
        
        p1_wins = 0
        p2_wins = 0
        avg_loss_window = []
        
        for episode in range(num_episodes):
            winner, avg_loss = play_game(agent1, agent2)
            avg_loss_window.append(avg_loss)
            
            if winner == PLAYER1:
                p1_wins += 1
            else:
                p2_wins += 1
            
            if episode % 100 == 0:
                win_rate = p1_wins / (p1_wins + p2_wins) * 100
                avg_loss_last_100 = np.mean(avg_loss_window[-100:])
                print(f"Episode {episode}, P1 Win Rate: {win_rate:.2f}%, Avg Loss: {avg_loss_last_100:.4f}")
                p1_wins = 0
                p2_wins = 0
        
        agent1.save_model('model1.h5')
        agent2.save_model('model2.h5')

if __name__ == "__main__":
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    print(f"Training on: {'/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'}")
    train_agents()