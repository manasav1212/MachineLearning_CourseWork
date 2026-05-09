import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Action encoding: 0=left, 1=right, 2=up, 3=down
# Moving "up" increases y; moving "right" increases x.
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = ['left', 'right', 'up', 'down']
# Index change for each action
ACTION_POSITION_CHANGES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
}

def load_bmp(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    # Pixels darker than 128 are obstacles (1), brighter are free (0)
    grid = (arr < 128).astype(np.int8)
    return grid


def abstract_map(grid, abs_size):
    rows, cols = grid.shape
    # This is the block size
    x_block = rows // abs_size
    y_block = cols // abs_size

    abstraction = np.zeros((abs_size, abs_size), dtype=np.int8)

    # Iterate over every cell of the grid
    for i in range(abs_size):
        for j in range(abs_size):
            # Compute the bounds of the original-pixel block that maps to (i, j)
            r_start = i * x_block
            c_start = j * y_block
            # Add remaining pixels to the last block
            r_end = (i + 1) * x_block if i < abs_size - 1 else rows
            c_end = (j + 1) * y_block if j < abs_size - 1 else cols
            block = grid[r_start:r_end, c_start:c_end]
            # Perform over-approximation: if any pixel in the block is an obstacle, mark obstacle
            # Easy numpy any since it is all 1 and 0
            if block.any():
                abstraction[i, j] = 1

    # Flip it vertically for visuals and more intuitive (x, y) plotting
    return np.flipud(abstraction)

class Environment:
    
    def __init__(self, grid, target, reward_strategy='S1'):
        self.grid = grid
        # numpy (rows, cols) = (height, width)
        self.height, self.width = grid.shape
        self.target = tuple(target)
        assert reward_strategy in ('S1', 'S2', 'S3'), "reward_strategy must be 'S1' or 'S2' or 'S3'"
        self.reward_strategy = reward_strategy

        tx, ty = self.target
        assert 0 <= tx < self.width and 0 <= ty < self.height, "target out of bounds"
        assert self.grid[ty, tx] == 0, "Target must be on a free cell"

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x, y):
        # Note: grid is indexed [y, x], not [x, y]
        return self.grid[y, x] == 1

    def is_free(self, x, y):
        return self.in_bounds(x, y) and not self.is_obstacle(x, y)

    def free_cells(self):
        """List of all (x, y) coordinates that are free cells."""
        y, x = np.where(self.grid == 0)
        return list(zip(x.tolist(), y.tolist()))

    def _reward_S1(self, x, y, hit_obstacle, reached_goal):
        """
        S1 Simple strategy.
        Big positive reward at goal, big negative at obstacle/out-of-bounds, 0 otherwise.
        """
        if reached_goal:
            return 100.0
        if hit_obstacle:
            return -100.0
        # No information for non-terminal moves: agent must learn purely from the
        # +100 / -100 signals.
        return 0.0

    def _reward_S2(self, x, y, hit_obstacle, reached_goal):
        """
        S2 strategy.
        Big positive at goal, big negative at obstacle.
        Small -1 per-step penalty which would mean shorter path is better.
        Distance-based: closer to goal = less penalty.
        """
        if reached_goal:
            return 100.0
        if hit_obstacle:
            return -100.0
        # Manhattan distance to goal
        tx, ty = self.target
        dist = abs(x - tx) + abs(y - ty)
        max_dist = self.width + self.height
        # cells far from goal get values near -1. Closer gets near 0.
        return -dist / max_dist
    
    def _reward_S3(self, x, y, hit_obstacle, reached_goal):
        """Step-penalty strategy: encourages shorter paths."""
        if reached_goal:
            return 100.0
        if hit_obstacle:
            return -100.0
        # Constant penalty per step
        return -1.0

    def _reward(self, x, y, hit_obstacle, reached_goal):
        if self.reward_strategy == 'S1':
            return self._reward_S1(x, y, hit_obstacle, reached_goal)
        elif self.reward_strategy == 'S2':
            return self._reward_S2(x, y, hit_obstacle, reached_goal)
        else:
            return self._reward_S3(x, y, hit_obstacle, reached_goal)

    def step(self, state, action):
        x, y = state
        x_change, y_change = ACTION_POSITION_CHANGES[action]
        new_x, new_y = x + x_change, y + y_change

        # Case 1: the move would go out of bounds or into an obstacle.
        # Agent stays in place, gets the obstacle penalty, but episode does NOT end.
        if (not self.in_bounds(new_x, new_y)) or self.is_obstacle(new_x, new_y):
            reward = self._reward(x, y, hit_obstacle=True, reached_goal=False)
            return (x, y), reward, False

        # Case 2: the action lands on the target. Episode will end here.
        if (new_x, new_y) == self.target:
            reward = self._reward(new_x, new_y, hit_obstacle=False, reached_goal=True)
            return (new_x, new_y), reward, True

        # Case 3: Normal action into a free cell.
        reward = self._reward(new_x, new_y, hit_obstacle=False, reached_goal=False)
        return (new_x, new_y), reward, False

    # We added the ax so that the evaluation can plot on the same axis without creating a new one
    def plot(self, ax=None, title=None, path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.grid, cmap='gray_r', origin='lower')
        
        # Draw the target as a red star
        tx, ty = self.target
        ax.plot(tx, ty, marker='*', color='red', markersize=18, label='target')

        # Plot the path
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '-', color='blue', linewidth=2, label='path')
            ax.plot(xs[0], ys[0], 'go', markersize=8, label='start')
            ax.legend()

        if title:
            ax.set_title(title)
        return ax

    
class Agent:
    def __init__(self, width, height, num_actions=4, alpha=0.1, gamma=0.5, epsilon=0.3):
        self.width = width
        self.height = height
        self.num_actions = num_actions
        # Hyperparameters to experiment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialized to zero
        self.q_table = np.zeros((width, height, num_actions), dtype=np.float64)

    def reset_Q(self):
        self.q_table = np.zeros((self.width, self.height, self.num_actions), dtype=np.float64)

    def choose_action(self, state, rng=None):
        if rng is None:
            rng = np.random
        x, y = state
        # If within epsilon, do exploration
        if rng.random() < self.epsilon:
            return rng.randint(self.num_actions)
        # Otherwise, pick the action with the highest current Q-value
        return int(np.argmax(self.q_table[x, y]))

    # After training, we select the best action without exploration during the final run
    def choose_best_action(self, state):
        x, y = state
        return int(np.argmax(self.q_table[x, y]))

    # Update q table using SARSA 
    def sarsa_update(self, s, a, r, s_next, a_next, done):
        x, y = s
        if done:
            target = r
        else:
            x_new, y_new = s_next
            target = r + self.gamma * self.q_table[x_new, y_new, a_next]
        # Update the Q-table
        self.q_table[x, y, a] += self.alpha * (target - self.q_table[x, y, a])

    # Update q table using Q-learning
    def qlearning_update(self, s, a, r, s_next, done):
        x, y = s
        if done:
            target = r
        else:
            x_new, y_new = s_next
            # The max is what makes this off-policy
            target = r + self.gamma * np.max(self.q_table[x_new, y_new])
        self.q_table[x, y, a] += self.alpha * (target - self.q_table[x, y, a])

def sarsa_train(env, agent, start_state=None, num_episodes=1000, max_steps=500, verbose=True, seed=None):
    rng = np.random.RandomState(seed)
    # Get all free cells, used for random episode starts
    free = env.free_cells()
    if env.target in free:
        free.remove(env.target)

    rewards_per_episode = []
    steps_per_episode = []

    iterator = range(num_episodes)
    if verbose:
        #  This is for the progress bar
        iterator = tqdm(iterator, desc='SARSA training')

    for ep in iterator:
        # Start random so that more states are covered during training. But if start_state is provided, use it instead.
        if start_state is None:
            s = free[rng.randint(len(free))]
        else:
            s = start_state

        # Pick the FIRST action before entering the loop
        a = agent.choose_action(s, rng=rng)

        ep_reward = 0.0
        ep_steps = 0

        #  Avoid infinite loops by limiting the number of steps per episode
        for _ in range(max_steps):
            # Take the action
            s_next, r, done = env.step(s, a)

            if done:
                # If done, then the episode is complete, so just update the q-table and break out of the steps loop to start the next episode
                agent.sarsa_update(s, a, r, s_next, a_next=None, done=True)
                ep_reward += r
                ep_steps += 1
                break

            # Pick the next action first, then use it in the update. Since it is SARSA, we choose based on epsilon
            a_next = agent.choose_action(s_next, rng=rng)
            agent.sarsa_update(s, a, r, s_next, a_next, done=False)

            # Update s and a for the next iteration
            s, a = s_next, a_next
            ep_reward += r
            ep_steps += 1

        rewards_per_episode.append(ep_reward)
        steps_per_episode.append(ep_steps)

    return rewards_per_episode, steps_per_episode

def qLearning_train(env, agent, start_state=None, num_episodes=1000, max_steps=500, verbose=True, seed=None):
    rng = np.random.RandomState(seed)
    free = env.free_cells()
    if env.target in free:
        free.remove(env.target)
    
    rewards_per_episode = []
    steps_per_episode = []
    iterator = range(num_episodes)
    
    if verbose:
        iterator = tqdm(iterator, desc='Q-Learning training')
    
    for ep in iterator:
        if start_state is None:
            s = free[rng.randint(len(free))]
        else:
            s = start_state

        ep_reward = 0.0
        ep_steps = 0
        
        for _ in range(max_steps):
            a = agent.choose_action(s, rng=rng)
            s_next, r, done = env.step(s, a)
            agent.qlearning_update(s, a, r, s_next, done)
            
            ep_reward += r
            ep_steps += 1
            s = s_next
            
            if done:
                break

        rewards_per_episode.append(ep_reward)
        steps_per_episode.append(ep_steps)
    return rewards_per_episode, steps_per_episode
    

# Plotting function
def plot_map(grid, target, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    height, width = grid.shape
    # extent makes the axis labels match cell coordinates (0..width, 0..height)
    ax.imshow(grid, cmap='gray_r', origin='lower', extent=[0, width, 0, height])
    tx, ty = target
    # +0.5 centers the marker inside the cell when extent is used
    ax.plot(tx + 0.5, ty + 0.5, marker='*', color='red',
            markersize=18, label='target')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()

#  Visualize the max q-table value for each cell
def plot_policy_arrows(env, agent, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    # Draw the map underneath the arrows
    env.plot(ax=ax, title=title)
    arrow_dx = {0: -0.4, 1: 0.4, 2: 0.0, 3: 0.0}
    arrow_dy = {0: 0.0, 1: 0.0, 2: 0.4, 3: -0.4}
    for (x, y) in env.free_cells():
        # Skip the arrow for target cell
        if (x, y) == env.target:
            continue
        a = agent.choose_best_action((x, y))
        ax.arrow(x, y, arrow_dx[a], arrow_dy[a],
                 head_width=0.2, head_length=0.15, fc='blue', ec='blue',
                 length_includes_head=True, alpha=0.6)
    return ax

def run_the_policy(env, agent, start, max_steps=500):
    s = tuple(start)
    # We will trace the path taken by the policy.
    path = [s]
    for _ in range(max_steps):
        # Always choose best action since this is evaluation i.e. only exploitation
        a = agent.choose_best_action(s)
        s_next, _, done = env.step(s, a)
        if done:
            path.append(s_next)
            return path, (s_next == env.target)
        # If the action didn't move us, the policy is stuck against a wall
        if s_next == s:
            return path, False
        path.append(s_next)
        # Detect osscillation: if we are going back and forth
        if len(path) >= 4 and path[-1] == path[-3] and path[-2] == path[-4]:
            return path, False
        s = s_next
    return path, False


def evaluate_policy(env, agent, max_steps=500):
    """
    Test accuracy is the fraction of free cells from which the greedy policy
    successfully reached the target.
    """
    free = env.free_cells()
    if env.target in free:
        free.remove(env.target)
    if len(free) == 0:
        return 0.0
    successes = 0
    for s in free:
        _, ok = run_the_policy(env, agent, s, max_steps=max_steps)
        if ok:
            successes += 1
    return successes / len(free)

import matplotlib.animation as animation

def animate_rollouts(env, agent, start_states, max_steps=200, interval=150, save_path=None, title=None):
    '''This is for the animation for visualizing the paths'''
    # Pre-calculate all paths
    runs = []
    for start in start_states:
        path, success = run_the_policy(env, agent, start, max_steps=max_steps)
        runs.append((start, path, success))
        print(f"Start {start}: {'reached goal' if success else 'failed'} in {len(path) - 1} steps")

    # Pick a different color for each run
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    # Flatten into a list of (run_index, step_index) for each frame
    frame_plan = []
    for ri, (_, path, _) in enumerate(runs):
        for si in range(len(path)):
            frame_plan.append((ri, si))

    fig, ax = plt.subplots(figsize=(7, 7))
    if title:
        # Title
        fig.suptitle(title, fontsize=14, fontweight='bold')

    def draw_frame(frame_idx):
        # Wipe the previous frame and redraw from scratch each time
        ax.clear()
        ri, si = frame_plan[frame_idx]
        start, path, _ = runs[ri]

        # Draw the base map
        env.plot(ax=ax, title=f"Run {ri + 1}/{len(runs)} from {start} — step {si}/{len(path) - 1}")

        # Draw all previously completed runs as faded trails
        for prev_ri in range(ri):
            _, prev_path, _ = runs[prev_ri]
            xs = [p[0] for p in prev_path]
            ys = [p[1] for p in prev_path]
            ax.plot(xs, ys, '-', color=colors[prev_ri],linewidth=2, alpha=0.4)
            ax.plot(xs[0], ys[0], 'o', color=colors[prev_ri], markersize=8, alpha=0.5)

        # Draw the current run up to this frame
        partial = path[:si + 1]
        xs = [p[0] for p in partial]
        ys = [p[1] for p in partial]
        ax.plot(xs, ys, '-', color=colors[ri], linewidth=2.5, alpha=0.9)
        # Start as colored dot
        ax.plot(xs[0], ys[0], 'o', color=colors[ri], markersize=10)
        # Current position as bigger dot for visibility
        ax.plot(xs[-1], ys[-1], 'o', color='orange', markersize=14, markeredgecolor='black', markeredgewidth=1.5)

    anim = animation.FuncAnimation( fig, draw_frame, frames=len(frame_plan), interval=interval, repeat=False)

    if save_path:
        fps = max(1, 1000 // interval)
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return anim