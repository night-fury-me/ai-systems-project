> _Repository for WRV/AISysProj/WS2425/A2.3 - Wumpus Quest/team639_

## **Topic:** WS24/25 Assignment 2.3: Wumpus Cave Navigation Agent


## Overview

This project implements an intelligent agent that navigates a hazardous Wumpus cave grid to collect gold and exit safely within 100 steps. The agent combines Markov Decision Process (MDP) principles with skill-based probabilistic reasoning to handle:

- Stochastic movement (80-10-10 direction variance)
- Wumpus combat with fighting skill checks
- Bridge navigation with agility-based success probabilities
- Time-constrained path planning

Key capabilities include dynamic skill allocation, loop prevention, and emergency return strategies.

---

## Solution Summary
A detailed technical explanation of the solution approach can be found in **[Solution_Summary.pdf](Solution_Summary.pdf)**.

---

## Dependencies

### Core Requirements
- **Python 3.11+** (Tested on 3.11.4)
- **No external libraries required** - Pure Python implementation


### Installation
```bash
git clone https://gitlab.rrze.fau.de/wrv/AISysProj/ws2425/a2.3-wumpus-quest/team639.git
cd team639
```

### How to Run
```bash
python agent.py agent-configs/ws2425-quest-1.json
```


### Repository Structure
```bash

├── agent.py                # Main agent logic (MDP implementation)
├── client.py               # Server communication handler
├── utils.py                # Helper functions (movement, parsing)
├── agent-configs/          # Skill allocation profiles
│   ├── ws2425-quest-1.json
│   ├── ws2425-quest-2.json
│   ├── ws2425-quest-3.json
│   └── ws2425-quest-4.json
├── assignment-2.3.pdf      # Problem specification
├── assignment-2.3-guide.pdf # Implementation guide
└── Solution_Summary.pdf    # Detailed solution documentation
```

### Key Features

- Adaptive Skill Allocation
    - Dynamically distributes 12 (changes based on environment) skill points between fighting/navigation
    - Weighted by Wumpus/bridge density in map

- Probabilistic Action Model
    - 80-10-10 movement variance handling
    - Monte Carlo skill success simulation (1000 samples)

- Anti-Loop Mechanisms
    - 5-step position memory with revisit penalties

### Important Notes

- Server Configuration
    - Ensure client.py has correct server credentials
    - Keep configuration files in agent-configs/ unchanged for graded tasks

- Heuristic Choices
    - Manhattan distance prioritization for gold/exit
    - Time penalty factor of 1000 per overdue step
    - Bridge risk threshold: P(success) > 0.6

- Execution Constraints
    - Max 100 steps per episode
    - Mandatory EXIT action from starting position

### Contact

Maintainer: Redwanul Karim
Email: [redwanul.karim@fau.de]

