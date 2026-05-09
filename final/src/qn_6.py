import time
import numpy as np
from lib import *
import pandas as pd

MAP_CONFIGS = [
    {
        'name': 'Map 1',
        'path': '../data/map1.bmp',
        'start_positions': [(0, 18), (15, 18), (18, 18), (0, 8), (5, 8), (10, 8), (15, 8), (18, 8)],
        'target': (18, 2),
    },
    {
        'name': 'Map 2',
        'path': '../data/map2.bmp',
        'start_positions': [(0, 38), (10, 38), (20, 38), (30, 38), (35, 38), (5, 15), (15, 15), (35, 15)],
        'target': (38, 2),
    },
    {
        'name': 'Map 3',
        'path': '../data/map3.bmp',
        'start_positions': [(0, 0), (3, 35), (12, 30), (25, 25), (35, 30), (10, 10), (20, 15), (35, 20)],
        'target': (38, 2),
    },
    {
        'name': 'Map 4',
        'path': '../data/map4.bmp',
        'start_positions': [(0, 38), (15, 38), (35, 38), (0, 25), (38, 18), (0, 5), (15, 0), (10, 15), (35, 10), (25, 24), (39, 23)],
        'target': (38, 2),
    },
]

ABS_SIZE = 40
ALPHA = 0.1
NUM_EPISODES = 10000
MAX_STEPS = 500
SEED = 0

# Default hyperparameters
DEFAULT_GAMMA = 0.5
DEFAULT_EPSILON = 0.5
DEFAULT_REWARD = 'S2'

# Map 4 is used for comparison B, C, D
MAP4_INDEX = 3


def build_env(map_cfg, reward_strategy):
    grid = load_bmp(map_cfg['path'])
    grid = abstract_map(grid, min(ABS_SIZE, grid.shape[1]))
    return Environment(grid, target=map_cfg['target'], reward_strategy=reward_strategy)


def run_one(map_cfg, policy_type, alpha, gamma, epsilon, reward, num_episodes=NUM_EPISODES, seed=SEED, animate=False):
    """Run one experiment and return a dict of metrics."""
    env = build_env(map_cfg, reward)
    agent = Agent(env.width, env.height, num_actions=4, alpha=alpha, gamma=gamma, epsilon=epsilon)

    if (policy_type == 'SARSA'):
        training_fxn = sarsa_train
    # Replace with 'Q-learning' training function. Assuming the function structure is similar to sarsa_train 
    else:
        training_fxn = qLearning_train

    start = time.perf_counter()
    rewards, steps = training_fxn(env, agent, start_state=None, num_episodes=num_episodes, max_steps=MAX_STEPS, seed=seed, verbose=False)
    time_elapsed = time.perf_counter() - start

    acc = evaluate_policy(env, agent, max_steps=MAX_STEPS)
    
    if animate:
        title = (f"{policy_type} — {map_cfg['name']} gamma = {gamma}, epsilon = {epsilon}, reward={reward}, acc={acc*100:.1f}%)")
        animate_rollouts(env, agent, start_states=map_cfg['start_positions'], max_steps=MAX_STEPS, interval=1, title=title)

    return {
        'policy_type': policy_type,
        'map': map_cfg['name'],
        'epsilon': epsilon,
        'gamma': gamma,
        'reward': reward,
        'time_s': time_elapsed,
        'episodes': num_episodes,
        'accuracy': acc,
        'final_avg_reward': float(np.mean(rewards[-50:])),
        'final_avg_steps': float(np.mean(steps[-50:])),
    }


def print_table(rows, columns, title):
    print("==============================================================")
    print(title)
    print("==============================================================")
    df = pd.DataFrame(rows)[columns]
    if 'accuracy' in df.columns:
        df['accuracy'] = df['accuracy'].apply(lambda v: f"{v * 100:.2f}%")
    for col in df.select_dtypes(include='float').columns:
        df[col] = df[col].round(3)
    print(df.to_string(index=False))


def compare_map_complexity(animate=False):
    print("Map Complexity comparison")
    rows = []
    for algo in ('SARSA', 'Q-learning'):
        for map_cfg in MAP_CONFIGS:
            print(f"  Running {algo} on {map_cfg['name']}")
            r = run_one(map_cfg, algo, ALPHA, DEFAULT_GAMMA, DEFAULT_EPSILON, DEFAULT_REWARD, animate=animate)
            rows.append(r)
    print_table(rows, ['policy_type', 'map', 'time_s', 'episodes', 'accuracy', 'final_avg_reward'],
                f' Map Complexity (gamma={DEFAULT_GAMMA}, epsilon={DEFAULT_EPSILON}, reward_strategy={DEFAULT_REWARD})'
    )
    return rows


def compare_exploration(animate=False):
    map_cfg = MAP_CONFIGS[MAP4_INDEX]
    print(f"Exploration Rate comparison ({map_cfg['name']}, gamma ={DEFAULT_GAMMA})")
    rows = []
    for policy in ('SARSA', 'Q-learning'):
        for eps in (0.0, 0.5, 1.0):
            print(f"  Running {policy} with epsilon ={eps}")
            r = run_one(map_cfg, policy, ALPHA, DEFAULT_GAMMA, eps, DEFAULT_REWARD, animate=animate)
            rows.append(r)
    print_table(rows, ['policy_type', 'epsilon', 'time_s', 'episodes', 'accuracy', 'final_avg_reward'],
                f'Exploration Rate ({map_cfg["name"]}, gamma={DEFAULT_GAMMA}, reward={DEFAULT_REWARD})'
    )
    return rows


def compare_discount_values(animate=False):
    map_cfg = MAP_CONFIGS[MAP4_INDEX]
    print(f"Discount Value comparison ({map_cfg['name']}, epsilon={DEFAULT_EPSILON})")
    rows = []
    for algo in ('SARSA', 'Q-learning'):
        for gamma in (0.1, 0.5, 1.0):
            print(f"  Running {algo} with gamma ={gamma}")
            r = run_one(map_cfg, algo, ALPHA, gamma, DEFAULT_EPSILON, DEFAULT_REWARD, animate=animate)
            rows.append(r)
    print_table(rows, ['policy_type', 'gamma', 'time_s', 'episodes', 'accuracy', 'final_avg_reward'],
                f'Discount Value ({map_cfg["name"]}, epsilon={DEFAULT_EPSILON}, reward={DEFAULT_REWARD})')
    return rows


def best_hyperparams(rows_B, rows_C, policy_type):
    # Since we already ran B and C, we can pick the best hyperparameters from those results
    eps_results = [r for r in rows_B if r['policy_type'] == policy_type]
    gamma_results = [r for r in rows_C if r['policy_type'] == policy_type]

    # Pick the row with the highest accuracy from each comparison
    best_eps_row = max(eps_results, key=lambda r: r['accuracy'])
    best_gamma_row = max(gamma_results, key=lambda r: r['accuracy'])

    return best_eps_row['epsilon'], best_gamma_row['gamma']


def compare_reward_strategies(rows_B, rows_C, animate=False):
    map_cfg = MAP_CONFIGS[MAP4_INDEX]
    print(f"Reward Strategy comparison ({map_cfg['name']}, best epsilon & gamma per algorithm)")
    rows = []
    for policy_type in ('SARSA', 'Q-learning'):
        eps, gamma = best_hyperparams(rows_B, rows_C, policy_type)
        print(f"  {policy_type}: best epsilon={eps}, best gamma ={gamma}")
        for reward in ('S1', 'S2'):
            print(f"Running {policy_type} with reward={reward}")
            r = run_one(map_cfg, policy_type, ALPHA, gamma, eps, reward, animate=animate)
            rows.append(r)
    print_table(rows, ['policy_type', 'epsilon', 'gamma', 'reward', 'time_s', 'episodes', 'accuracy', 'final_avg_reward'],
                f'Reward Strategy ({map_cfg["name"]}, best epsilon & gamma per algorithm)')
    return rows


#  Change animate = True to visualize the test runs
rows_A = compare_map_complexity(animate=False)
rows_B = compare_exploration(animate=False)
rows_C = compare_discount_values(animate=False)
rows_D = compare_reward_strategies(rows_B, rows_C, animate=False)