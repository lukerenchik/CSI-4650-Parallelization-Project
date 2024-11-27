# main.py
from game import HangmanGame
from ai_player import AIPlayer

def main():
    print("Welcome to Hangman!")
    player = AIPlayer()

    game = HangmanGame(player)
    game.play()

if __name__ == '__main__':
    main()
