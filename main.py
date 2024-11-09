# main.py
from game import HangmanGame
from player import Player
from ai_player import AIPlayer

def main():
    while True:
        print("Welcome to Hangman!")
        mode = input("Choose player type - Human (H) or AI (A): ").strip().lower()
        if mode == 'h':
            player = Player()
        elif mode == 'a':
            player = AIPlayer()
        else:
            print("Invalid choice. Please enter 'H' for Human or 'A' for AI.")
            continue

        game = HangmanGame(player)
        game.play()

        play_again = input("Do you want to play again? (Y/N): ").strip().lower()
        if play_again != 'y':
            print("Thank you for playing Hangman!")
            break

if __name__ == '__main__':
    main()
