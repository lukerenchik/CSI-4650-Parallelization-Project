# AI Hangman: Demonstrating GPU Acceleration and Parallelism

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Features](#features)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Steps](#steps)
5. [Usage](#usage)
   - [Running the Non-Parallel Training Script](#running-the-non-parallel-training-script)
   - [Running the Parallel Training Script](#running-the-parallel-training-script)
6. [Results and Findings](#results-and-findings)
   - [Training Time Comparison](#training-time-comparison)
     - [Non-Parallel Training on CPU](#non-parallel-training-on-cpu)
     - [Non-Parallel Training on GPU](#non-parallel-training-on-gpu)
   - [Observations](#observations)
   - [Potential Findings](#potential-findings)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)
9. [References](#references)

## Introduction

This project implements an AI Hangman game using PyTorch, designed to showcase the advantages of GPUs and parallelism in machine learning tasks. By comparing training times and model performance between CPU and GPU executions, as well as between serial and parallel data generation, we demonstrate how leveraging these technologies can significantly accelerate computation.

## Project Overview

The project consists of two main scripts:

- **Non-Parallel Training Script (`train_model.py`)**: Trains the Hangman AI model without parallel data generation.
- **Parallel Training Script (`train_model_parallel.py`)**: Utilizes multiprocessing to generate training data in parallel, and leverages GPU acceleration if available.

Both scripts train an LSTM-based neural network (`HangmanLSTM` from `new_ai_player.py`) to predict the next letter in a game of Hangman, based on the current state of the word and the letters already guessed.

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

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ai-hangman.git
   cd ai-hangman
   

2. **Install Dependencies**
   ```bash
   pip install torch numpy
   
3. **Prepare the Word List**
    - Ensure you have a words.txt in the project directory with a list of words, one per line.

## Usage

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Running the Non-Parallel Training Script

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Running the Parallel Training Script

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Results & Findings

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Training Time Comparison

1. **Non-Parallel on CPU**
2. **Parallel on CPU**
3. **Non-Parallel on GPU**
4. **Parallel on GPU**

### Observations

### Potential Findings

## Conclusion

This project effectively demonstrates how GPUs and parallelism can accelerate machine learning tasks.
By comparing the training times and performance of the non-parallel and parallel scripts on CPU and GPU, we observe:

- Significant reduction in training time when using GPUs.
- Potential for further acceleration with parallel data generation.
- Importance of leveraging available hardware resources to optimize machine learning workflows.