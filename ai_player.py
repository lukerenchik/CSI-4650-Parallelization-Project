import os
import string
import numpy as np
import psutil
import torch
import torch.nn as nn
import json
class AIPlayer:
    def __init__(self, model_path, train=False):

        self.filename = f"{model_path.rstrip('.pth')}_cpu_usage.json"
        self.clear_json()

        self.all_letters = set(string.ascii_lowercase)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HangmanLSTM().to(self.device)
        if not train:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.guessed_letters = set()
        self.guess_number = 0

    def reset(self):
        self.guessed_letters = set()

    def guess(self, guessed_letters, word_display):
        self.guessed_letters = guessed_letters
        self.guess_number += 1
        word_vector = self.encode_word_state(word_display)
        guessed_vector = self.encode_guessed_letters()

        word_tensor = torch.tensor(word_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        guessed_tensor = torch.tensor(guessed_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(word_tensor, guessed_tensor).squeeze(0).cpu().numpy()

        # Penalize already guessed letters
        for letter in guessed_letters:
            output[ord(letter) - ord('a')] = -1

        # Choose the letter with the highest probability
        guess = chr(np.argmax(output) + ord('a'))
        print(f"AI guesses: {guess}")

        cpu_usage = self.calculate_cpu_usage()
        
        self.write_cpu_usage(cpu_usage)
        return guess

    def encode_word_state(self, word_display):
        word_vector = np.zeros((len(word_display), 27))  # 27: 26 letters + 1 for '_'
        for i, char in enumerate(word_display):
            if char == '_':
                word_vector[i, 26] = 1
            elif 'a' <= char <= 'z':
                word_vector[i, ord(char) - ord('a')] = 1
        return word_vector

    def encode_guessed_letters(self):
        guessed_vector = np.zeros(26)
        for letter in self.guessed_letters:
            guessed_vector[ord(letter) - ord('a')] = 1
        return guessed_vector

    def calculate_cpu_usage(self):

        return {
            '1s': psutil.cpu_percent(interval=1),
            '2s': psutil.cpu_percent(interval=2),
            '5s': psutil.cpu_percent(interval=5),
            '10s': psutil.cpu_percent(interval=10)
        }

    def write_cpu_usage(self, cpu_usage):
        data = {
            'guess_number': self.guess_number,
            'cpu_usage': cpu_usage
        }

        with open(self.filename, "a") as json_file:
            json.dump(data, json_file)
            json_file.write("\n")

    def clear_json(self):

        if os.path.exists(self.filename):
            with open(self.filename, 'w') as json_file:
                json_file.truncate(0) 
class HangmanLSTM(nn.Module):
    def __init__(self):
        super(HangmanLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=27, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128 + 26, 64)
        self.fc2 = nn.Linear(64, 26)

    def forward(self, word_input, guessed_input):
        lstm_out, _ = self.lstm(word_input)
        lstm_last = lstm_out[:, -1, :]  # Last output from LSTM
        combined = torch.cat((lstm_last, guessed_input), dim=1)
        x = torch.relu(self.fc1(combined))
        return torch.softmax(self.fc2(x), dim=-1)