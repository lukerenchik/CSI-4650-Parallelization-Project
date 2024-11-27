from game import HangmanGame
from ai_player import AIPlayer
from graph import plot
from train_model import Train_Model
import time

def main():

    # Hardcoded word so they both run on the same word. Can change word 
    hangman_word = "army"

    model_key = "modelPath"
    num_worker_key = "numWorkers"

    players = [{model_key: "slower_hangman_model.pth", num_worker_key: 2},
                {model_key: "faster_hangman_model.pth", num_worker_key: 8}]
    
    for player in players:
        start_time = time.time()

        Train_Model(player[model_key], player[num_worker_key])
        player = AIPlayer(player[model_key])

        game = HangmanGame(player, hangman_word)
        game.play()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time Played: {elapsed_time} seconds")

    plot(players[0][model_key], players[1][model_key])

if __name__ == '__main__':
    main()
