from lib import *
import time

ABS_SIZE = 40
SEED = 42

MAP_CONFIGS = [
    {
        'name': 'Map 1',
        'path': '../data/map1.bmp',
        'target': (18, 2)
    },
    {
        'name': 'Map 2',
        'path': '../data/map2.bmp',
        'target': (38, 10)
    },
    {
        'name': 'Map 3',
        'path': '../data/map3.bmp',
        'target': (28, 5)
    },
    {
        'name': 'Map 4',
        'path': '../data/map4.bmp',
        'target': (38, 10)
    },
]

for cfg in MAP_CONFIGS:
    map_name = cfg['name']
    grid = load_bmp(cfg['path'])
    grid = abstract_map(grid, min(grid.shape[1], ABS_SIZE))
    target = cfg['target']
    env1 = Environment(grid, target)
    print(f'Environment reward strategy: {env1.reward_strategy}, target: {target}, width: {env1.width}, height: {env1.height}')
    env1.plot(title=map_name)
    
    env2 = Environment(grid, target, reward_strategy='S2')
    print(f'Environment reward strategy: {env2.reward_strategy}, target: {target}, width: {env2.width}, height: {env2.height}')
    