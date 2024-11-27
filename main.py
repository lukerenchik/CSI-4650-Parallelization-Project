from game import HangmanGame
from ai_player import AIPlayer
from train_model import Train_Model
import time

def main():

    hangman_word = "army"
    players = [{"modelPath": "slower_hangman_model.pth", "numWorkers": 2},
                {"modelPath": "faster_hangman_model.pth", "numWorkers": 8}]
    
    for player in players:
        start_time = time.time()

        Train_Model(player["modelPath"], player["numWorkers"])
        player = AIPlayer(player["modelPath"])

        game = HangmanGame(player, hangman_word)
        game.play()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time Played: {elapsed_time} seconds")

if __name__ == '__main__':
    main()
