# AI Hangman: Demonstrating GPU Acceleration and Parallelism

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Features](#features)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Steps](#steps)
5. [Usage](#usage)
6. [Conclusion](#conclusion)


## Introduction

This project implements an AI Hangman game using PyTorch, designed to showcase the advantages of GPUs and parallelism in machine learning tasks. By comparing training times and model performance between CPU and GPU executions, as well as between serial and parallel data generation, we demonstrate how leveraging these technologies can significantly accelerate computation.

## Project Overview

The project consists of two main scripts:

- **Non-Parallel Training Script (`train_model_non_parallelize.py`)**: Trains the Hangman AI model without parallel data generation.
- **Parallel Training Script (`train_model_parallelize.py`)**: Utilizes multiprocessing to generate training data in parallel, and leverages GPU acceleration if available.
- **words.txt**: The text file includes all the input data for training.
- **new_ai_player.py**: The ai model for playing the hangman game, which includes the LSTM model that will be import for training.
-  **hangman.ipynb**: The file will be used for training in colab in a GPU setting, which includes non-parallel and parallel models, and their respective outputs.
-  **hangman_cpu.ipynb**: The file will be used for training in colab in a CPU setting, which includes non-parallel and parallel models, and their respective outputs.

Both scripts train an LSTM-based neural network (`HangmanLSTM` from `new_ai_player.py`) to predict the next letter in a game of Hangman, based on the current state of the word and the letters already guessed. In both hangman.ipynb and hangman_cpu.ipynb files, there are the full description of how parallel and non-parallel models work.

## Features

- **AI Hangman Player**: Uses an LSTM neural network to predict the next best letter to guess.
- **Parallel Data Generation**: Speeds up the data preparation phase using Python's `multiprocessing` module.
- **GPU Acceleration**: Utilizes CUDA-compatible GPUs to accelerate training with PyTorch.
- **Performance Comparison**: Provides insights into the benefits of parallelism and GPU usage by comparing training times and losses.

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch (with CUDA support for GPU acceleration)
- NumPy
- We utilized CUDA 12.4 & associated Torch version

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/lukerenchik/CSI-4650-Parallelization-Project/
   cd CSI-4650-Parallelization-Project/
   

2. **Install Dependencies**
   
   If Using Local HW:

   - Verify NVIDIA CUDA Driver:
   
   ```bash
   nvidia-smi
   ```
   
   - To check CUDA toolkit version, run
   
   ```bash 
   nvcc --version
   ```
   
   Find matching torch wheel on: https://pytorch.org/
   
   - Install Remaining Dependency
   ```bash
   pip install numpy
   ```
   
   On Colab:
   ```bash
   pip install torch numpy
   ```
   
4. **Prepare the Word List**
    - Ensure you have a words.txt in the project directory with a list of words, one per line.


## Usage

- CPU (Non-paralellized): train_model_non_paralellize.py

- CPU (Paralellized): train_model_non_paralellize.py

- GPU: GPU_Parallelize.py
  
- Colab: hangman.ipynb and hangman_cpu.ipynb (upload the "words.txt" and "new_ai_player.py" to Colab and make sure they are saved, then run each shell to see their respective outputs)

## Conclusion

This project effectively demonstrates how GPUs and parallelism can accelerate machine learning tasks.
By comparing the training times and performance of the non-parallel and parallel scripts on CPU and GPU, we observe:

- Significant reduction in training time when using GPUs.
- Potential for further acceleration with parallel data generation,especially for CPU setting.
- Importance of leveraging available hardware resources to optimize machine learning workflows.
