# game.py
import random

class HangmanGame:
    def __init__(self, player):
        self.player = player

    def load_words(self):
        # Load words from words.txt
        try:
            with open('words.txt', 'r') as file:
                words = [line.strip().lower() for line in file if line.strip()]
            return words
        except FileNotFoundError:
            print("Error: words.txt file not found.")
            exit(1)

    def play(self):
        self.word_list = self.load_words()
        self.word = random.choice(self.word_list)
        self.word_letters = set(self.word)
        self.guessed_letters = set()
        self.lives = 6  # Number of allowed mistakes

        while self.lives > 0 and not self.word_letters.issubset(self.guessed_letters):
            print(f"\nYou have {self.lives} lives left.")
            word_display = ''.join(
                [letter if letter in self.guessed_letters else '_' for letter in self.word]
            )
            print('Word: ' + ' '.join(word_display))

            guess = self.player.guess(self.guessed_letters, word_display)

            if guess in self.word_letters:
                if guess in self.guessed_letters:
                    print(f"You have already guessed '{guess}'.")
                else:
                    print(f"Good guess: {guess}")
                    self.guessed_letters.add(guess)
            else:
                if guess in self.guessed_letters:
                    print(f"You have already guessed '{guess}'.")
                else:
                    print(f"Wrong guess: {guess}")
                    self.lives -= 1
                    self.guessed_letters.add(guess)

        if self.word_letters.issubset(self.guessed_letters):
            print(f"\nCongratulations! You guessed the word: {self.word}")
        else:
            print(f"\nGame over! The word was: {self.word}")
