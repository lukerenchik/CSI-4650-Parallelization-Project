# main.py
import argparse
from game import HangmanGame
from ai_player import AIPlayer
from train_model import Train_Model

def main(model_filename, num_workers, hangman_word):

    Train_Model(model_filename, num_workers)
    player = AIPlayer(model_filename)

    game = HangmanGame(player, hangman_word)
    game.play()

if __name__ == '__main__':

    # Passing arguments through CLI
    parser = argparse.ArgumentParser(description="Welcome to Hangman!")
    parser.add_argument('model_filename', type=str, help="Path to training data")
    parser.add_argument('num_workers', type=int, help="The numbers of workers")
    parser.add_argument('hangman_word', type=str, help="Word to be used in game")
    
    # Get argumetns
    args = parser.parse_args()

    main(args.model_filename, args.num_workers, args.hangman_word)
