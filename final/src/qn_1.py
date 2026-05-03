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
        'target': (38, 2)
    },
    {
        'name': 'Map 3',
        'path': '../data/map3.bmp',
        'target': (38, 2)
    },
    {
        'name': 'Map 4',
        'path': '../data/map4.bmp',
        'target': (38, 2)
    },
]

for cfg in MAP_CONFIGS:
    map_name = cfg['name']
    grid = load_bmp(cfg['path'])
    grid = abstract_map(grid, min(grid.shape[1], ABS_SIZE))
    target = cfg['target']
    plot_map(grid, target=target, title=map_name)
    