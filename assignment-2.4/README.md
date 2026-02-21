> _Repository for WRV/AISysProj/WS2425/A2.4 - Guess the Word/team639_

## **Topic:** WS24/25 Assignment 2.4: Guess the Word

## Overview

This project implements an intelligent agent for a word-guessing game where words are drawn from a predefined list of cities. The agent is designed to efficiently guess the correct word while minimizing the number of steps using:

- **Probabilistic Word Filtering** (TZ vs. non-TZ bias)
- **Shannon Entropy-Based Letter Selection** (Information Gain)
- **Early Word Guessing** (Threshold-based)
- **Dynamic Letter Candidate Generation** (Adaptive letter selection)
- **Advanced Partial-Reveal Handling** (One occurrence revealed per letter)

---

## Solution Summary
A detailed technical explanation of the solution approach can be found in **[Solution_Summary.pdf](Solution_Summary.pdf)**.

---

## Dependencies

### Core Requirements
- **Python 3.11+** (Tested on 3.11.4)
- **Pandas (latest)**
- **No other external libraries required** - Pure Python implementation

### Installation
```bash
git clone https://gitlab.rrze.fau.de/wrv/AISysProj/ws2425/a2.4-guess-the-word/team639.git
cd team639
```

### How to Run
```bash
python agent.py simple-env.json
```

### Repository Structure
```bash
├── advanced-env.json            # Config file for advanced environment
├── agent.py                     # Main agent logic
├── client.py                    # Server communication handler
├── simple-env.json              # Config file for advanced environment
├── Solution_Summary.pdf         # Detailed solution documentation
├── utils.py                     # Utility functions to load csv dataset
└── worldcities.csv.bz2          # csv dataset to cities and their population
```

### Key Features

- **Dynamic Candidate Generation**
    - Letter candidates are generated dynamically based on letter frequency in the remaining possible words.
    - Avoids relying on a fixed priority order like "ABCDEFGH...".

- **Information Gain for Letter Selection**
    - Computes Shannon entropy-based **expected information gain** for each letter.
    - Maximizes partitioning effectiveness.

- **Early Word Guessing**
    - If a word’s probability exceeds a threshold (e.g., 50%), the agent guesses the word instead of a letter.
    - Reduces unnecessary intermediate guesses.

- **Advanced Partial-Reveal Handling**
    - For words where a guessed letter appears multiple times, only **one** occurrence is revealed per guess.
    - The agent correctly partitions word probabilities accordingly.

### Important Notes

- **Word Probability Initialization**
    - 50% of words are from Tanzanian cities, 50% from non-Tanzanian cities.
    - Each word starts with equal probability within its group.

- **Guessing Strategy**
    - If only **one** possible word remains, guess it immediately.
    - If a word has **high probability**, guess it early.
    - Otherwise, choose the letter **with maximum information gain**.

### Contact

Maintainer: Redwanul Karim  
Email: [redwanul.karim@fau.de]
