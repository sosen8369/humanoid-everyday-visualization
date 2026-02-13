import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize

from config import EPSILON, COLORMAP

def normalize(group, global_fmax):
    medians = group.median(axis=0)
    f = (group - medians).abs()
    res = f.copy()
    
    z_score = (group - group.mean(axis=0)) / (group.std(axis=0) + EPSILON)
    scores = z_score.apply(lambda x: np.mean(np.abs(np.diff(x))))
    
    mask_smooth = scores <= 0.6
    mask_vibrate = ~mask_smooth
    
    if mask_smooth.any():
        denom_smooth = f.loc[:, mask_smooth].apply(lambda col: col.nlargest(3).mean()) + EPSILON
        res.loc[:, mask_smooth] = f.loc[:, mask_smooth] / denom_smooth
        
    if mask_vibrate.any():
        res.loc[:, mask_vibrate] = f.loc[:, mask_vibrate] / global_fmax
        
    return res

def value2color(val, vmin, vmax):
    norm_tool = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps[COLORMAP]
    rgba = cmap(norm_tool(val))
    return (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

def load_tactile(folder, episode):
    pattern = os.path.join('**', folder, '**', f'episode_{episode}', 'data.json')
    path = glob.glob(pattern, recursive=True)
    if not path:
        raise FileNotFoundError(f'No files found for {folder}/{episode}/data.json')
    
    with open(path[0]) as f:
        data = json.load(f)
    if not data:
        raise RuntimeError(f'No valid data found in the files for episode {folder}/episode_{episode}/data.json')
    
    sensors = defaultdict(list)
    for frame in data:
        for sensor in frame['states']['hand_pressure_state']:
            s_id = sensor['sensor_id']
            sensors[s_id].append(sensor['usable_readings'])

    return pd.concat(
        {key: pd.DataFrame(val) for key, val in sensors.items()}, 
        names=['ID', 'frame']
    )