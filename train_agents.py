import numpy as np
import os
from q_learning_agent import QLearningAgent, get_available_device
from game_environment import get_reward
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY
from collections import deque
import random

def train_agent_with_self_play(agent, num_episodes=10000, save_interval=1000, load_path=None):
    if load_path and os.path.exists(load_path):
        agent.load_model(load_path)
        print(f"Loaded model from {load_path}")
    
    historical_models = deque(maxlen=10)  # Queue to store past models
    initial_weights = agent.model.get_weights()
    historical_models.append(initial_weights)

    for episode in range(num_episodes):
        game = Game()  # Initialize a new game
        done = False
        
        # Randomly choose between current model and a historical model
        if len(historical_models) > 1 and np.random.random() < 0.5:
            opponent_weights = historical_models[np.random.randint(len(historical_models))]
            agent.model.set_weights(opponent_weights)
        
        # Randomly decide if the agent goes first or second
        agent_goes_first = np.random.choice([True, False])
        if not agent_goes_first:
            game.current_player = PLAYER2
        
        while not done:
            if game.current_player == agent.player:
                state = agent.get_state(game)
                action = agent.get_best_move(game)
                
                # Apply the move
                if len(action) == 2:
                    game.place_checker(*action)
                else:
                    game.move_checker(*action)
                
                next_state = agent.get_state(game)
                winner = game.check_winner()
                
                if winner != EMPTY:
                    done = True
                    reward = 1 if winner == agent.player else -1
                else:
                    reward = get_reward(game, agent.player)
                
                loss = agent.train(state, action, reward, next_state, done)
            else:
                # Simulate opponent's move (random for simplicity)
                opponent_moves = agent.get_possible_moves(game)
                opponent_move = random.choice(opponent_moves)
                if len(opponent_move) == 2:
                    game.place_checker(*opponent_move)
                else:
                    game.move_checker(*opponent_move)
                
                winner = game.check_winner()
                if winner != EMPTY:
                    done = True
            
            game.current_player = PLAYER1 if game.current_player == PLAYER2 else PLAYER2
        
        # Every 50 episodes, save the current model weights
        if (episode + 1) % 50 == 0:
            historical_models.append(agent.model.get_weights())
            agent.model.set_weights(initial_weights)  # Reset to most recent model
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss if 'loss' in locals() else 'N/A'}")
        
        # Save the model periodically
        if (episode + 1) % save_interval == 0:
            save_path = f"model_episode_{episode+1}.h5"
            agent.save_model(save_path)
            print(f"Saved model to {save_path}")

    return agent

# Use this function to train your agent
agent = QLearningAgent(PLAYER1)
trained_agent = train_agent_with_self_play(agent, num_episodes=10000, save_interval=1000, load_path="last_saved_model.h5")

# Save the final model
trained_agent.save_model("final_model.h5")