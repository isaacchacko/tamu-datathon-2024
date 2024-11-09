from q_learning_agent import QLearningAgent
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY

def play_game(agent1, agent2):
    game = Game()
    done = False
    
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
        
        current_agent.train(state, move, reward, next_state, done)
        
        game.current_player *= -1
    
    return winner

def train_agents(num_episodes=10000):
    agent1 = QLearningAgent(PLAYER1)
    agent2 = QLearningAgent(PLAYER2)
    
    for episode in range(num_episodes):
        winner = play_game(agent1, agent2)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Winner: {'Player 1' if winner == PLAYER1 else 'Player 2'}")
    
    agent1.save_model('model1.h5')
    agent2.save_model('model2.h5')

if __name__ == "__main__":
    train_agents()