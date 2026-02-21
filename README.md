# AI Systems Project — Course Assignments (WS24/25)

This repository contains solution code and documentation for multiple assignments from the WS24/25 AI Systems course (`Author: Redwanul Karim`).

Overview
--------
- assignment-2.3: Wumpus Cave Navigation Agent (MDP + skill reasoning)
- assignment-2.4: Guess the Word (entropy-based word/letter selection)
- assignment-2.5: Math Publication Classification (hierarchical transformer)

Each assignment folder contains a self-contained implementation, README, and solution summary. This top-level README summarizes the contents, dependencies, and how to run each assignment.

Repository structure (high level)
---------------------------------

project-root/
- assignment-2.3/   # Wumpus quest agent
- assignment-2.4/   # Guess-the-word agent
- assignment-2.5/   # Math-document classification (transformer)

For details, see the README inside each assignment folder.

Quick start by assignment
-------------------------

Assignment 2.3 — Wumpus Cave Navigation
- Location: assignment-2.3/
- Purpose: an intelligent agent to navigate a stochastic Wumpus cave, allocate skill points, fight Wumpuses and collect gold within 100 steps.
- Key files:
	- agent.py — main agent logic (MDP heuristics, movement model)
	- client.py — server protocol implementation and run loop
	- utils.py — helper functions (grid parsing, survival probabilities, movement)
	- agent-configs/ — sample skill allocation JSON profiles
	- Solution_Summary.pdf — full approach description
- Dependencies: Python 3.11+ (pure Python; no external packages required)
- Run locally (example):

```bash
python assignment-2.3/agent.py assignment-2.3/agent-configs/ws2425-quest-1.json
```

Notes:
- The agent reads the initial configuration from the provided JSON profile and communicates with the server via `client.py`.
- See assignment-2.3/README.md for algorithmic details and heuristics (anti-loop, skill allocation, bridge handling).

Assignment 2.4 — Guess the Word
- Location: assignment-2.4/
- Purpose: letter-selection / word-guessing agent using expected information gain (Shannon entropy) and probabilistic filtering over a city-name dataset.
- Key files:
	- agent.py — agent class implementing the strategy (uses `client.Agent` wrapper)
	- utils.py — dataset loader & helper functions (loads worldcities dataset)
	- simple-env.json / advanced-env.json — example environment configs
	- worldcities.csv.bz2 — dataset (cities)
	- Solution_Summary.pdf — full writeup
- Dependencies: Python 3.11+, pandas
- Run locally (example):

```bash
python assignment-2.4/agent.py assignment-2.4/simple-env.json
```

Notes:
- The agent computes expected information gain for candidate letters and can guess a full word early when its posterior probability exceeds a threshold.
- The code supports an "advanced" environment variant with modified reveal rules — see the README and solution summary.

Assignment 2.5 — Math Publication Classification
- Location: assignment-2.5/
- Purpose: hierarchical transformer that encodes formulas and documents for arXiv category classification.
- Key files:
	- model.py — FormulaTransformer, DocumentTransformer, MathDocClassifier
	- train.py — training loop (PyTorch, MLflow logging, schedulers)
	- evaluate.py — evaluation utilities
	- generate_result.py — result generation helper
	- data/ — dataset processing and tokenizer (`data/data_processing.py`, `data/math_symbol_extractor.py`)
	- Dockerfile — image for training and MLflow server
	- server_interaction.py — server-facing classification entrypoint
	- Solution_Summary.pdf — full writeup
- Dependencies: PyTorch (2.6+), tokenizers (Hugging Face), mlflow. CUDA-enabled Docker recommended for GPU runs.
- Typical workflow using Docker (recommended):

```bash
# Build image
docker build -f assignment-2.5/Dockerfile -t ai-sys-img .

# Prepare dataset (inside container)
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace ai-sys-img python assignment-2.5/data/data_processing.py

# Train
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace ai-sys-img python assignment-2.5/train.py

# Evaluate
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace ai-sys-img python assignment-2.5/evaluate.py assignment-2.5/Data/example-test-results.json assignment-2.5/Data/my_test_result.json
```

Or run training directly if the environment has the required Python packages and GPU available:

```bash
python assignment-2.5/train.py
```

Notes and important configuration
--------------------------------
- assignment-2.3: the agent expects configuration JSON files in `assignment-2.3/agent-configs/`.
- assignment-2.4: the city dataset is `assignment-2.4/worldcities.csv.bz2` and `pandas` is required for loading.
- assignment-2.5: `data/mathml_tokenizer.json` and preprocessed training/test pickles are used by `train.py` and `server_interaction.py`. MLflow tracking default URI used in code: `http://localhost:5000`.

Developer notes
---------------
- Python version: the assignments were tested with Python 3.11.4 (see individual READMEs).
- Virtual environment recommended. Example (Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create one per project if needed
```

- For assignment-2.5, if you plan to run training locally, install `torch`, `tokenizers`, `mlflow`, `tqdm`, `scikit-learn` and other dependencies listed in the script headers.

What I included in this README
------------------------------
- High-level overview of each assignment
- Key files and dataset pointers
- Short run instructions and Docker commands for assignment 2.5

Missing items / suggestions
---------------------------
- No LICENSE file detected; add one if you plan to publish or share externally.
- A unified `requirements.txt` at repo root would help replicate environments quickly. Each assignment can also include its own requirements file.
- Small convenience scripts (Makefile or top-level scripts) to build and run Docker commands would simplify reproducibility.

Contact
-------
Maintainer: Redwanul Karim
Email: redwanul.karim@fau.de

