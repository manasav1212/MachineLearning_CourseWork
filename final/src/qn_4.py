from lib import *
import time

ABS_SIZE = 40
TARGET = (38, 2)

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.3

NUM_EPISODES = 10000
MAX_STEPS = 500
SEED = 42

MAP_CONFIGS = [
    {
        'name': 'Map 1',
        'path': '../data/map1.bmp',
        'start_positions': [(0, 18), (15, 18), (18, 18), (0, 8), (5, 8), (10, 8), (15, 8), (18, 8)],
        'target': (18, 2)
    },
    {
        'name': 'Map 2',
        'path': '../data/map2.bmp',
        'start_positions': [(0, 38), (10, 38), (20, 38), (30, 38), (35, 38), (5, 15), (15, 15), (35, 15),],
        'target': (38, 2)
    },
    {
        'name': 'Map 3',
        'path': '../data/map3.bmp',
        'start_positions': [(0, 0), (3, 35), (12, 30), (25, 25), (35, 30), (10, 10), (20, 15), (35, 20)],
        'target': (38, 2)
    },
    {
        'name': 'Map 4',
        'path': '../data/map4.bmp',
        'start_positions': [(0, 38), (15, 38), (35, 38), (0, 25), (38, 18), (0, 5), (15, 0), (10, 15), (35, 10), (25, 24), (39, 23)],
        'target': (38, 2)
    },
]

REWARD_STRATEGIES = ['S1', 'S2', 'S3']

results = {}

for cfg in MAP_CONFIGS:
    map_name = cfg['name']
    grid = load_bmp(cfg['path'])
    grid = abstract_map(grid, min(grid.shape[1], ABS_SIZE))
    target = cfg.get('target', TARGET)

    for strategy in REWARD_STRATEGIES:
        label = f"{map_name} | reward={strategy}"
        print('========================================================')
        print(f"Training: {label}")
        print(f"  gamma={GAMMA}, epsilon={EPSILON}, alpha={ALPHA}, episodes={NUM_EPISODES}, max_steps={MAX_STEPS}")
        print('========================================================')

        env = Environment(grid, target, reward_strategy=strategy)
        agent = Agent(env.width, env.height, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

        t0 = time.time()
        rewards, steps = sarsa_train(env, agent, start_state=None, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, seed=SEED)
        elapsed = time.time() - t0

        print(f"  training time:           {elapsed:.2f} s")
        print(f"  final 50-ep avg reward:  {np.mean(rewards[-50:]):.2f}")
        print(f"  final 50-ep avg steps:   {np.mean(steps[-50:]):.1f}")

        results[(map_name, strategy)] = {
            'rewards': rewards,
            'steps': steps,
            'env': env,
            'agent': agent,
            'start_positions': cfg['start_positions'],
            'training_time': elapsed,
        }

# Evaluating and visualization
for (map_name, strategy), data in results.items():
    rewards = data['rewards']
    env = data['env']
    agent = data['agent']
    start_positions = data['start_positions']

    label = f"{map_name} | reward={strategy}"
    print(f"\n============= {label} ================")

    acc = evaluate_policy(env, agent, max_steps=MAX_STEPS)
    print(f"  Test accuracy: {acc * 100:.2f}%")

    # Learning curve + policy arrows. Uncomment for the graphs.
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # axes[0].plot(rewards, alpha=0.3, label='per-episode')
    # window = 50
    
    # if len(rewards) >= window:
    #     rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    #     axes[0].plot(np.arange(window - 1, len(rewards)), rolling, label=f'{window}-ep rolling mean')
    
    # axes[0].set_xlabel('Episode')
    # axes[0].set_ylabel('Total reward')
    # axes[0].set_title(f'SARSA learning curve — {label}')
    # axes[0].legend()
    # axes[0].grid(alpha=0.3)

    # plot_policy_arrows(env, agent, ax=axes[1], title=f'SARSA policy — {label} (acc = {acc * 100:.1f}%)')

    # plt.tight_layout()
    # plt.show()

    # Visualize the paths from this map's specific starting positions
    print(f"  Animating rollouts from {len(start_positions)} starting positions...")
    anim = animate_rollouts(env, agent, start_positions, max_steps=MAX_STEPS, interval=1)