import subprocess
import time
import os

def run_judge_engine():
    result = subprocess.run(['python3', 'judge_engine.py'], capture_output=True, text=True)
    output = result.stdout.split('\n')
    winner = None
    game_string = None
    for line in reversed(output):
        if line.startswith("Winner:"):
            winner = 'PLAYER1' if 'PLAYER1' in line else 'PLAYER2'
        if line.startswith("Game String:"):
            game_string = line.split(": ")[1]
        if winner and game_string:
            break
    return winner, game_string, '\n'.join(output)

def calculate_win_rate(num_games=100):
    player1_wins = 0
    player2_wins = 0
    errors = 0

    log_file = f"game_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    for i in range(num_games):
        print(f"Running game {i+1}/{num_games}")
        winner, game_string, full_output = run_judge_engine()
        
        with open(log_file, 'a') as f:
            f.write(f"Game {i+1}\n")
            f.write(full_output)
            f.write("\n" + "="*50 + "\n\n")

        if winner == 'PLAYER1':
            player1_wins += 1
        elif winner == 'PLAYER2':
            player2_wins += 1
        else:
            errors += 1
        
        # Add a small delay to prevent overwhelming the system
        time.sleep(0.1)

    print(f"\nResults after {num_games} games:")
    print(f"Player 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Errors: {errors}")
    print(f"\nPlayer 1 win rate: {player1_wins/num_games:.2%}")
    print(f"Player 2 win rate: {player2_wins/num_games:.2%}")
    print(f"Error rate: {errors/num_games:.2%}")

    with open(log_file, 'a') as f:
        f.write(f"\nFinal Results:\n")
        f.write(f"Player 1 wins: {player1_wins}\n")
        f.write(f"Player 2 wins: {player2_wins}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Player 1 win rate: {player1_wins/num_games:.2%}\n")
        f.write(f"Player 2 win rate: {player2_wins/num_games:.2%}\n")
        f.write(f"Error rate: {errors/num_games:.2%}\n")

    print(f"\nDetailed game log written to {log_file}")

if __name__ == "__main__":
    num_games = int(input("Enter the number of games to run: "))
    calculate_win_rate(num_games)