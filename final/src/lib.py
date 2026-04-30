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
    grid = (arr < 128).astype(np.int8)
    return grid


def abstract_map(grid, abs_size):
    rows, cols = grid.shape
    # This is the block size
    x_block = rows // abs_size
    y_block = cols // abs_size

    abstraction = np.zeros((abs_size, abs_size), dtype=np.int8)

    for i in range(abs_size):
        for j in range(abs_size):
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

    # Flip it for visuals and more intuitive (x, y) indexing
    return np.flipud(abstraction)

class Environment:
    
    def __init__(self, grid, target, reward_strategy='S1'):
        self.grid = grid
        self.height, self.width = grid.shape
        self.target = tuple(target)
        assert reward_strategy in ('S1', 'S2'), "reward_strategy must be 'S1' or 'S2'"
        self.reward_strategy = reward_strategy

        # Validate that the target is on a free cell within boundary
        tx, ty = self.target
        assert 0 <= tx < self.width and 0 <= ty < self.height, "target out of bounds"
        assert self.grid[ty, tx] == 0, "target must be on a free cell"

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x, y):
        return self.grid[y, x] == 1

    def is_free(self, x, y):
        return self.in_bounds(x, y) and not self.is_obstacle(x, y)

    def free_cells(self):
        """List of all (x, y) coordinates that are free (will be used for evaluation)."""
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
        return 0.0

    def _reward_S2(self, x, y, hit_obstacle, reached_goal):
        """
        S2 strategy.
        Big positive at goal, big negative at obstacle.
        Small per-step penalty which would mean shorter path is better.
        Distance-based: closer to goal = less penalty.
        """
        if reached_goal:
            return 100.0
        if hit_obstacle:
            return -100.0
        # Block distance to goal
        tx, ty = self.target
        dist = abs(x - tx) + abs(y - ty)
        max_dist = self.width + self.height
        # Per-step cost is -1, plus a small bonus that grows as we approach goal. But the reward is always negative
        return -1.0 + (1.0 - dist / max_dist)

    def _reward(self, x, y, hit_obstacle, reached_goal):
        if self.reward_strategy == 'S1':
            return self._reward_S1(x, y, hit_obstacle, reached_goal)
        else:
            return self._reward_S2(x, y, hit_obstacle, reached_goal)

    def step(self, state, action):
        x, y = state
        x_change, y_change = ACTION_POSITION_CHANGES[action]
        new_x, new_y = x + x_change, y + y_change

        # Out of bounds or it hit the obstacle.
        if (not self.in_bounds(new_x, new_y)) or self.is_obstacle(new_x, new_y):
            reward = self._reward(x, y, hit_obstacle=True, reached_goal=False)
            return (x, y), reward, False

        # Reached the goal.
        if (new_x, new_y) == self.target:
            reward = self._reward(new_x, new_y, hit_obstacle=False, reached_goal=True)
            return (new_x, new_y), reward, True

        # Other free cell. Depends on the reward strategy
        reward = self._reward(new_x, new_y, hit_obstacle=False, reached_goal=False)
        return (new_x, new_y), reward, False

    def plot(self, title=None, path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(self.grid, cmap='gray_r', origin='lower')
        
        tx, ty = self.target
        ax.plot(tx, ty, marker='*', color='red', markersize=18, label='target')

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '-', color='blue', linewidth=2, label='path')
            ax.plot(xs[0], ys[0], 'go', markersize=8, label='start')
            ax.legend()

        if title:
            ax.set_title(title)
        plt.show()

    
class Agent:
    def __init__(self, width, height, num_actions=4,
                 alpha=0.1, gamma=0.5, epsilon=0.3):
        self.width = width
        self.height = height
        self.num_actions = num_actions
        # Hyperparameters to experiment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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
        # Else we do exploitation
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
        self.q_table[x, y, a] += self.alpha * (target - self.q_table[x, y, a])

    # Update q table using Q-learning
    def qlearning_update(self, s, a, r, s_next, done):
        x, y = s
        if done:
            target = r
        else:
            x_new, y_new = s_next
            target = r + self.gamma * np.max(self.q_table[x_new, y_new])
        self.q_table[x, y, a] += self.alpha * (target - self.q_table[x, y, a])