import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from new_ai_player import HangmanLSTM

def generate_training_data(word_list, num_samples, max_word_length=10):
    """Generate training data based on word list."""
    print("Generating training data...")
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
                for _ in range(3):  # Repeat to balance the dataset
                    data.append((word_input, guessed_input, target_letter))
            else:
                data.append((word_input, guessed_input, target_letter))
    print(f"Generated {len(data)} training samples.")
    return data

def encode_word_state(word_display, max_word_length):
    """Encode word state as a one-hot matrix."""
    word_vector = np.zeros((max_word_length, 27))  # 27: 26 letters + 1 for '_'
    for i, char in enumerate(word_display[:max_word_length]):
        if char == '_':
            word_vector[i, 26] = 1  # Represent blanks as the 27th feature
        elif 'a' <= char <= 'z':
            word_vector[i, ord(char) - ord('a')] = 1
    return word_vector

def encode_guessed_letters(guessed_letters):
    """Encode guessed letters as a one-hot vector."""
    guessed_vector = np.zeros(26)
    for letter in guessed_letters:
        guessed_vector[ord(letter) - ord('a')] = 1
    return guessed_vector

def train_model(word_list, model_path='hangman_model.pth', num_samples=10000, epochs=25, batch_size=32, lr=0.001):
    """Train the HangmanLSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate training data
    data = generate_training_data(word_list, num_samples)

    # Prepare tensors
    print("Preparing data tensors...")
    inputs_word = torch.tensor(np.array([item[0] for item in data]), dtype=torch.float32)
    inputs_guessed = torch.tensor(np.array([item[1] for item in data]), dtype=torch.float32)
    targets = torch.tensor(np.array([item[2] for item in data]), dtype=torch.long)

    # Create DataLoader
    print("Creating DataLoader...")
    dataset = torch.utils.data.TensorDataset(inputs_word, inputs_guessed, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = HangmanLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for word_batch, guessed_batch, target_batch in dataloader:
            word_batch = word_batch.to(device)
            guessed_batch = guessed_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(word_batch, guessed_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # Load the word list from file
    words_file = "words.txt"
    try:
        with open(words_file, 'r') as f:
            word_list = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(word_list)} words from {words_file}.")
    except FileNotFoundError:
        print(f"Error: {words_file} not found.")
        exit(1)

    # Train the model
    print("Starting training...")
    train_model(word_list, model_path="hangman_model.pth", num_samples=20000, epochs=25, batch_size=32, lr=0.001)
    print("Training completed.")
