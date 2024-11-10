# train_model.py

from strategic_agent import NEATAgent
import os

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'neat_config.txt')
    
    # Initialize NEAT agent with configuration file
    agent = NEATAgent(config_path)
    
    # Train for 1000 generations (adjust as needed)
    agent.train(generations=1000)
    
    print("Training complete.")