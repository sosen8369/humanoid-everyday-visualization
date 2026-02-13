import os
import glob
import argparse
import subprocess

import cv2
import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    SENSOR_MAPPING, INVALID_VALUE, STANDARD_VALUE,
    FPS, WIDTH, HEIGHT, GRAPH_HEIGHT, EPSILON, BACKGROUND_PATH
)
from utils import normalize, value2color, load_tactile
from ShapeGroup import PyramidGroup, GridGroup

def draw_barh(current_norm, width, height, xlim):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    plot_values, plot_labels = [], []
    for s_id, row in current_norm.sort_index().iterrows():
        for s_idx, val in row.items():
            if not np.isnan(val):
                plot_values.append(val)
                plot_labels.append(f"{s_id}_{s_idx}")

    ax.barh(plot_labels, plot_values, color='skyblue')
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()
    ax.set_title("Normalized Tactile Profile", fontsize=10)
    ax.tick_params(axis='y', labelsize=6)
    
    plt.tight_layout()
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba() #type: ignore
    img = np.asarray(rgba_buffer)
    img = img[:, :, :3]
    plt.close(fig)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def draw_squares(tactile_data, mapper, base_img, vmin, vmax):
    img = base_img.copy()
    groups = {
        0: (GridGroup, 195, 230, 25, 45),
        1: (PyramidGroup, 160, 90, 25, 0),
        2: (GridGroup, 330, 455, 25, 20),
        3: (PyramidGroup, 460, 540, 25, 120),
        4: (GridGroup, 430, 365, 25, 20),
        5: (PyramidGroup, 565, 455, 25, 120),
        6: (GridGroup, 295, 300, 25, 45),
        7: (GridGroup, 250, 335, 25, 45),
        8: (GridGroup, 205, 370, 25, 45)
    }

    for s_id, group in mapper.items():
        datas = tactile_data.loc[s_id]
        group_id, indexs = group
        colors: list[tuple[int, int, int] | None] = [None]*len(indexs)
        valid = [False]*len(indexs)

        for v, target_pos in zip(datas, indexs):
            if v >= vmin: 
                colors[target_pos] = value2color(v, vmin, vmax)
                valid[target_pos] = True
            else:
                colors[target_pos] = None
                valid[target_pos] = False

        gtype, x, y, size, angle = groups[group_id]
        g = gtype(x, y, size, colors, valid, angle)
        g.draw(img)
    return img

def visualize_tactile(base_img, tactile_data, width, height, vmin, vmax):    
    left_img = draw_squares(tactile_data, SENSOR_MAPPING['left'], base_img, vmin, vmax)
    # ---
    cv2.putText(left_img, '8', (190,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '9', (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '1', (240, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '2', (205, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '3', (320, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '5', (450, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '7', (320, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '4', (590, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(left_img, '6', (480, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # ---
    left_img = cv2.resize(left_img, (width, height))
    right_img = draw_squares(tactile_data, SENSOR_MAPPING['right'], base_img, vmin, vmax)
    right_img = cv2.flip(right_img, 1)
    # ---
    cv2.putText(right_img, '17', (620 - 190,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '18', (620 - 220, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '10', (620 - 240, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '11', (620 - 205, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '12', (620 - 320, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '14', (620 - 450, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '16', (620 - 320, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '13', (620 - 590, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(right_img, '15', (620 - 480, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # ---
    right_img = cv2.resize(right_img, (width, height))

    return np.hstack([left_img, right_img])


def render_video(folder, episode, output_path, fps = FPS):
    tactile_raw = load_tactile(folder, episode)
    tactile_cleaned = tactile_raw.copy()

    tactile_cleaned = tactile_cleaned.replace(INVALID_VALUE, np.nan)

    means_all = tactile_cleaned.groupby(level='ID').mean()
    f_all = (tactile_cleaned.sub(means_all, level='ID')).abs()
    
    high_mask = means_all >= STANDARD_VALUE
    
    avg_dev_per_id = f_all.groupby(level='ID').mean()
    
    if high_mask.any().any():
        global_scale_val = avg_dev_per_id[high_mask].max().max() + EPSILON
    else:
        global_scale_val = avg_dev_per_id.max().max() + EPSILON

    tactile_norm = tactile_cleaned.groupby(level='ID', group_keys=False).apply(
        normalize, global_fmax=global_scale_val
    )

    global_max_val = tactile_norm.max().max()
    graph_xlim = max(1.2, global_max_val * 1.1)

    norm_min = 0.0 
    norm_max = tactile_norm.max().max()

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH*2, HEIGHT+GRAPH_HEIGHT))
    
    base_img = cv2.imread(BACKGROUND_PATH)
    pattern = os.path.join('**', folder, '**', f'episode_{episode}', 'color', '*.jpg')
    img_paths = glob.glob(pattern, recursive=True)

    lengths = [
        len(img_paths),
        tactile_raw.index.get_level_values('frame').max() + 1
    ]
    
    for f_idx in tqdm.trange(min(lengths)):
        img_color = cv2.imread(img_paths[f_idx])
        img_color = cv2.resize(img_color, (WIDTH, HEIGHT)) if img_color is not None else np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        frame_tactile = tactile_norm.xs(f_idx, level='frame')
        img_graph = draw_barh(frame_tactile, WIDTH, HEIGHT, graph_xlim)
        
        img_top = np.hstack([img_color, img_graph])

        if f_idx in tactile_norm.index.get_level_values('frame'):
            frame_tactile = tactile_norm.xs(f_idx, level='frame')
            
            img_bottom = visualize_tactile(
                base_img, frame_tactile, WIDTH, GRAPH_HEIGHT, 
                vmin=norm_min, vmax=norm_max
            )
        else:
            img_bottom = np.zeros((GRAPH_HEIGHT, WIDTH*2, 3), dtype=np.uint8)

        out.write(np.vstack([img_top, img_bottom]))

    out.release()
    print(f'finished at {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--folder', type=str, required=True)
    parser.add_argument('-e', '--episode', type=int, default=0)
    parser.add_argument('-o', '--out', type=str, default='output.mp4')
    parser.add_argument('-z', '--zip', action='store_true')

    args = parser.parse_args()

    if args.zip:
        folder = args.folder
        render_video(
            folder=folder,
            episode=args.episode,
            output_path=args.out
        )

        tmp = subprocess.run([
            'ffmpeg', '-y', '-i', args.out,
            '-vcodec', 'libx264', '-crf', '23',
            '-pix_fmt', 'yuv420p', args.out + 'tmp.mp4'
        ], check=True, capture_output=True, text=True)
        os.remove(args.out)
        os.rename(args.out + 'tmp.mp4', args.out)

    else:
        render_video(
            folder=args.folder,
            episode=args.episode,
            output_path=args.out
        )

"""
Usage:
python main.py \
  --folder click_a_keyboard_key --episode 0 \
  --out click_a_keyboard_key0.mp4 --zip
"""