# ai_player.py
import string
import random
import numpy as np

class AIPlayer:
    def __init__(self):
        self.all_letters = set(string.ascii_lowercase)
        self.model = self.load_model()
        self.reset()

    def reset(self):
        self.guessed_letters = set()

    def load_model(self):
        # Load a pre-trained machine learning model if available
        # For this example, we'll use English letter frequencies
        letter_frequencies = {
            'e': 12.70, 't': 9.06, 'a': 8.17, 'o': 7.51,
            'i': 6.97, 'n': 6.75, 's': 6.33, 'h': 6.09,
            'r': 5.99, 'd': 4.25, 'l': 4.03, 'c': 2.78,
            'u': 2.76, 'm': 2.41, 'w': 2.36, 'f': 2.23,
            'g': 2.02, 'y': 1.97, 'p': 1.93, 'b': 1.49,
            'v': 0.98, 'k': 0.77, 'x': 0.15, 'j': 0.15,
            'q': 0.10, 'z': 0.07
        }
        return letter_frequencies

    def guess(self, guessed_letters, word_display):
        self.guessed_letters = guessed_letters

        possible_letters = list(self.all_letters - self.guessed_letters)

        if not possible_letters:
            # All letters have been guessed
            guess = random.choice(list(self.all_letters))
        else:
            # Use the model to predict the next letter
            probabilities = np.array([self.model.get(letter, 0) for letter in possible_letters])
            probabilities /= probabilities.sum()  # Normalize to sum to 1
            guess = np.random.choice(possible_letters, p=probabilities)

        print(f"AI guesses: {guess}")
        return guess
