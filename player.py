class Player:
    def guess(self, guessed_letters, word_display):
        while True:
            guess = input("Guess a letter: ").lower()
            if len(guess) != 1 or not guess.isalpha():
                print("Please enter a single alphabetical character.")
            elif guess in guessed_letters:
                print("You have already guessed that letter. Try again.")
            else:
                return guess
