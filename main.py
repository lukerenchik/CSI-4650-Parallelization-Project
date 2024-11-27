# main.py
import argparse
from game import HangmanGame
from ai_player import AIPlayer

def main(model_filename, num_workers):

    player = AIPlayer(model_filename, num_workers)

    game = HangmanGame(player)
    game.play()

if __name__ == '__main__':

    # Passing arguments through CLI
    parser = argparse.ArgumentParser(description="Welcome to Hangman!")
    parser.add_argument('model_filename', type=str, help="Path to training data")
    parser.add_argument('num_workers', type=int, help="The numbers of workers")
    
    # Get argumetns
    args = parser.parse_args()

    main(args.model_filename, args.num_workers)
