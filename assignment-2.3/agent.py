"""
    To use this implementation, you simply have to implement `agent_function` such that it returns a legal action.
    You can then let your agent compete on the server by calling
        python3 example_agent.py path/to/your/config.json

    You can interrupt the script at any time and continue at a later time.
    The server will remember the actions you have sent.

    Note:
        Once your agent "works", you can set `parallel_runs` to True.
        Then the server simulates multiple games in parallel and bundles requests.
        This reduces the overhead from waiting for the server.
        Furthermore, if you set `processes=5`, the client will use multiple processes
        for the calls to your agent_function.
"""

import math
import random
import utils
from collections import deque


def agent_function(req_dict, _info):

    # print('Received request:', json.dumps(req_dict, indent=2))

    if not req_dict['history']:
        grid    = [list(row) for row in req_dict['map'].split('\n')]
        count_w = sum(row.count('W') for row in grid)
        count_b = sum(row.count('B') for row in grid)
        
        total_points        = req_dict.get("free-skill-points", 0)
        total_challenges    = count_w + count_b
        
        if total_challenges == 0:
            return {"agility": 0, "fighting": total_points}
        else:
            fighting    = round((count_w / total_challenges) * total_points)
            agility     = total_points - fighting
            fighting    = max(0, min(total_points, fighting))
            agility     = total_points - fighting
            return {"agility": agility, "fighting": fighting}


    grid    = [list(row) for row in req_dict['map'].split('\n')]
    history = req_dict['history']
    skills  = req_dict['skill-points']

    count_w = sum(row.count('W') for row in grid)
    count_b = sum(row.count('B') for row in grid)
    count_p = sum(row.count('P') for row in grid)
    count__ = sum(row.count(' ') for row in grid)

    odd_ratio = (count_w + count_b + count_p) / (count_b + count_w + count_p + count__)

    if odd_ratio > 0.30:
        return 'EXIT'

    start_pos, gold_positions, wumpus_positions = utils.parse_grid(grid)
    current_pos, collected_gold, killed_wumpuses, recent_positions = utils.parse_history(history, start_pos)
    remaining_gold  = gold_positions - {tuple(e['outcome']['collected-gold-at']) for e in history if 'collected-gold-at' in e.get('outcome', {})}
    steps_remaining = 100 - len(history)

    if current_pos == start_pos and collected_gold > 0:
        return 'EXIT'
    
    current_cell = grid[current_pos[1]][current_pos[0]]
    if current_cell == 'W' and current_pos not in killed_wumpuses:
        return 'FIGHT'
    
    possible_actions    = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    best_action         = 'EXIT'
    best_value          = -math.inf

    for action in possible_actions:
        expected_value = 0
        outcomes = utils.get_movement_outcomes(current_pos, action, grid)    
        
        next_pos = utils.move(current_pos, action)
        recent_pos, pos_freq = recent_positions
        
        revisited_pos = recent_pos.count(next_pos) > 0
        revisit_count = pos_freq.get(next_pos, 0)

        if revisited_pos and revisit_count > 3:
            expected_value -= 200 * revisit_count

        if not utils.is_valid(utils.move(current_pos, action), grid):
            continue

        for outcome, prob in outcomes:
            next_cord = utils.move(current_pos, outcome)
            next_cell = grid[next_cord[1]][next_cord[0]]
            cell_value = 0

            if next_cell == 'B' or next_cell == 'W':
                skill = skills.get('agility', 0) if next_cell == 'B' else skills.get('fighting', 0)
                threshold = 10 if  next_cell == 'B' else 13
                success_prob    = utils.get_survival_probability(skill, threshold)
                
                if success_prob < 0.60:
                    cell_value = -math.inf

            elif next_cell == 'P':
                cell_value = -math.inf
                
            cell_value += evaluate_cell_value(
                outcome, grid, remaining_gold, killed_wumpuses,
                start_pos, skills, steps_remaining - 1, recent_positions
            )

            if outcome == current_pos:
                cell_value = -100

            expected_value += prob * cell_value        

        if expected_value > best_value:
            best_value  = expected_value
            best_action = action

    if steps_remaining <= utils.manhattan(current_pos, start_pos) + 3 or best_action == 'EXIT':
        best_action = utils.get_direction(current_pos, start_pos, grid)
        if not best_action:
            valid_actions = [a for a in possible_actions if utils.is_valid(move(current_pos, a), grid)]
            best_action = random.choice(valid_actions) if valid_actions else 'EXIT'

    return best_action



def evaluate_cell_value(pos, grid, remaining_gold, killed_wumpuses, start_pos, skills, steps_left, recent_positions):
    if not utils.is_valid(pos, grid):
        return -math.inf

    cell_type = grid[pos[1]][pos[0]]
    value = 0

    recent_pos, pos_freq = recent_positions
    revisited_pos = recent_pos.count(pos) > 0
    revisit_count = pos_freq.get(pos, 0)

    revisit_penalty = 50 * revisit_count if revisited_pos and revisit_count > 3 else 0

    if cell_type == 'G' and pos in remaining_gold:
        value += 1000 
    elif cell_type == 'P':
        return -math.inf  
    elif cell_type == 'W' and pos not in killed_wumpuses:
        fight_skill     = skills.get('fighting', 0)
        success_prob    = utils.get_survival_probability(fight_skill, 13)
        value           = success_prob * 500 - (1 - success_prob) * 1000 

    gold_dist       = min([utils.manhattan(pos, g) for g in remaining_gold], default=0)
    start_dist      = utils.manhattan(pos, start_pos)
    time_penalty    = max(0, start_dist - steps_left) * 1000

    return value - gold_dist * 1000 - start_dist * 20 - time_penalty - revisit_penalty



if __name__ == '__main__':
    try:
        from client import run
    except ImportError:
        raise ImportError('You need to have the client.py file in the same directory as this file')

    import logging
    logging.basicConfig(level=logging.INFO)

    import sys
    config_file = sys.argv[1]

    run(
        # path to config file for the environment (in your personal repository)
        config_file,
        agent_function,
        # higher values will call the agent function on multiple requests in parallel (requires parallel_runs=True)
        processes=1,
        # stop after 1000 runs (then the rating is "complete")
        run_limit=1000,
        # set to True to let the server simulate multiple games in parallel
        parallel_runs=True,
        # set to True to "give up" all current games when you start the script.
        abandon_old_runs=True
    )
