import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from new_ai_player import HangmanLSTM  # Import my LSTM model
import random
import numpy as np

def generate_training_data(word_list, num_samples, max_word_length=10):
    """
    Generate synthetic training data for the Hangman game.

    Args:
        word_list (list): List of words to sample from.
        num_samples (int): Number of training samples to generate.
        max_word_length (int): Maximum length of the word representation.

    Returns:
        list: A list of tuples (word_input, guessed_input, target_letter).
    """
    data = []
    for _ in range(num_samples):
        word = random.choice(word_list)
        word_letters = set(word)
        guessed_letters = set()
        obscured_word = ['_' for _ in word]

        while len(guessed_letters) < len(word_letters):
            next_letter = random.choice(list(word_letters - guessed_letters))
            guessed_letters.add(next_letter)

            for i, char in enumerate(word):
                if char in guessed_letters:
                    obscured_word[i] = char

            word_input = encode_word_state(''.join(obscured_word), max_word_length)
            guessed_input = encode_guessed_letters(guessed_letters)
            target_letter = ord(next_letter) - ord('a')

            if '_' in ''.join(obscured_word):
                for _ in range(3):  # Increase representation for intermediate states
                    data.append((word_input, guessed_input, target_letter))
            else:
                data.append((word_input, guessed_input, target_letter))

    return data


def encode_word_state(word_display, max_word_length):
    """
    Encode the current word display into a vector representation.

    Args:
        word_display (str): The current display of the word.
        max_word_length (int): The maximum length of the word.

    Returns:
        np.ndarray: A 2D array representation of the word state.
    """
    word_vector = np.zeros((max_word_length, 27))  # 26 letters + 1 for '_'
    for i, char in enumerate(word_display[:max_word_length]):
        if char == '_':
            word_vector[i, 26] = 1
        elif 'a' <= char <= 'z':
            word_vector[i, ord(char) - ord('a')] = 1
    return word_vector


def encode_guessed_letters(guessed_letters):
    """
    Encode the guessed letters into a vector representation.

    Args:
        guessed_letters (set): The set of guessed letters.

    Returns:
        np.ndarray: A 1D array representation of guessed letters.
    """
    guessed_vector = np.zeros(26)
    for char in guessed_letters:
        guessed_vector[ord(char) - ord('a')] = 1
    return guessed_vector


# Limit PyTorch to a single thread
torch.set_num_threads(1)

def train_model(word_list, model_path="hangman_model_non_parallel.pth", num_samples=20000, epochs=25, batch_size=32, lr=0.001):
    device = torch.device("cpu")  # Force CPU usage
    model = HangmanLSTM().to(device)

    print("Generating training data...")
    start_time = time.time()
    data = generate_training_data(word_list, num_samples)
    end_time = time.time()
    print(f"Training data generation completed in {end_time - start_time:.2f} seconds.")

    print("Preparing data tensors...")
    inputs_word = torch.tensor(np.array([item[0] for item in data]), dtype=torch.float32)
    inputs_guessed = torch.tensor(np.array([item[1] for item in data]), dtype=torch.float32)
    targets = torch.tensor(np.array([item[2] for item in data]), dtype=torch.long)

    dataset = TensorDataset(inputs_word, inputs_guessed, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for no parallelism

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch_word, batch_guessed, batch_targets in dataloader:
            batch_word, batch_guessed, batch_targets = (
                batch_word.to(device),
                batch_guessed.to(device),
                batch_targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(batch_word, batch_guessed)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_end = time.time()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_end - epoch_start:.2f} seconds")

    total_training_time = time.time() - start_time
    print(f"Total training time (non-parallelized): {total_training_time:.2f} seconds")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    word_list = ["example", "hangman", "player", "training"]  # Replace with my actual word list later
    train_model(word_list)
