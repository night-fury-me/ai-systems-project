import math
import random
from collections import deque


def get_movement_outcomes(pos, action, grid):
    outcomes = {}
    for dir, prob in [
        (action, 0.8),
        (turn_left(action), 0.1),
        (turn_right(action), 0.1)
    ]:
        new_pos = move(pos, dir)
        if not is_valid(new_pos, grid):
            new_pos = pos  
        outcomes[new_pos] = outcomes.get(new_pos, 0.0) + prob
    return list(outcomes.items())


def get_direction(current, target, grid):
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    
    if abs(dx) > abs(dy):
        primary     = 'EAST' if dx > 0 else 'WEST'
        secondary   = 'SOUTH' if dy > 0 else 'NORTH'
    else:
        primary     = 'SOUTH' if dy > 0 else 'NORTH'
        secondary   = 'EAST' if dx > 0 else 'WEST'
    
    valid_directions = []
    
    for direction in [primary, secondary]:
        new_pos = move(current, direction)
        if is_valid(new_pos, grid):
            valid_directions.append(direction)
    
    if not valid_directions:
        for direction in ['EAST', 'WEST', 'SOUTH', 'NORTH']:
            new_pos = move(current, direction)
            if is_valid(new_pos, grid):
                valid_directions.append(direction)

    return valid_directions[0] if valid_directions else None

def parse_grid(grid):
    start_pos, golds, wumpuses = None, set(), set()
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            cell = grid[y][x]
            if cell == 'S':
                start_pos = (x, y)
            elif cell == 'G':
                golds.add((x, y))
            elif cell == 'W':
                wumpuses.add((x, y))
    return start_pos, golds, wumpuses

def parse_history(history, start_pos):
    current_pos, gold, killed = start_pos, 0, set()
    position_freq = {}
    recent_position = deque(maxlen=5)
    for entry in history:
        outcome = entry.get('outcome', {})
        if 'position' in outcome:
            current_pos = tuple(outcome['position'])
            if not current_pos in position_freq:
                position_freq[current_pos] = 0 
            position_freq[current_pos] += 1
            recent_position.append(current_pos)
        if 'collected-gold-at' in outcome:
            gold += 1
        if 'killed-wumpus-at' in outcome:
            killed.add(tuple(outcome['killed-wumpus-at']))
    return current_pos, gold, killed, tuple([recent_position, position_freq])

def get_survival_probability(skill_points, threshold, samples=1000):
    if skill_points < 3:
        return 0.0
    successes = 0
    for _ in range(samples):
        dice = [random.randint(1, 6) for _ in range(skill_points)]
        dice.sort(reverse=True)
        if sum(dice[:3]) >= threshold:
            successes += 1
    return successes / samples

def move(pos, action):
    x, y = pos
    if action == 'NORTH':
        return (x, y-1)
    elif action == 'SOUTH':
        return (x, y+1)
    elif action == 'EAST':
        return (x+1, y)
    elif action == 'WEST':
        return (x-1, y)
    return pos

def turn_left(action):
    return {'NORTH': 'WEST', 'WEST': 'SOUTH', 'SOUTH': 'EAST', 'EAST': 'NORTH'}.get(action, action)

def turn_right(action):
    return {'NORTH': 'EAST', 'EAST': 'SOUTH', 'SOUTH': 'WEST', 'WEST': 'NORTH'}.get(action, action)

def is_valid(pos, grid):
    x, y = pos
    return 0 <= y < len(grid) and 0 <= x < len(grid[y]) and grid[y][x] != 'X' and grid[y][x] != 'P'

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])