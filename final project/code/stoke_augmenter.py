#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
np.random.seed(0)


def strokes_to_lines(strokes):
    """Convert stroke−3 format to polyline format. """
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        x += float(strokes[i, 0])
        y += float(strokes[i, 1])
        line.append([x, y])
        if strokes[i, 2] == 1:
            lines.append(line)
            line = []
    return lines


def show_one_sample(strokes, save=False, filename=None, linewidth=10):
    fig = plt.figure(figsize=(5, 5))
    lines = strokes_to_lines(strokes)
    for idx in range(0, len(lines)):
        x = [x[0] for x in lines[idx]]
        y = [y[1] for y in lines[idx]]
        plt.plot(x, y, 'k-', linewidth=linewidth)
    ax = fig.gca()
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    ax.set_aspect(1)
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    if save:
        plt.axis("off")
        fig.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=13.3)
        plt.close(fig)


def move_strokes(seq: np.ndarray):
    result = copy.deepcopy(seq)
    for i in range(len(result)):
        if i == 0 or result[i-1][2] == 1:
            result[i][0] += np.random.randn() * 1.5
            result[i][1] += np.random.randn() * 1.5
    return result


def move_points(seq: np.ndarray):
    result = copy.deepcopy(seq)
    for i in range(len(result)):
        if i != 0 and result[i-1][2] != 1:
        # if True:
            result[i][0] += np.random.randn() * 0.01
            result[i][1] += np.random.randn() * 0.01
    return result


with open('./oracle_fs/seq/char_to_idx.txt') as f:
    idx2chr = list(f.readline())
chr2idx = {}
for i, c in enumerate(idx2chr):
    chr2idx[c] = i

shot = 1
img_target_dir = f"./oracle_fs/img/oracle_200_{shot}_shot_strokes/train"
seq_source_dir = f"./oracle_fs/seq/oracle_200_{shot}_shot/"

MULTIPLE = 30
for idx, char in enumerate(tqdm(idx2chr)):
    img_dir = os.path.join(img_target_dir, char)
    npzfile = os.path.join(seq_source_dir, f"{idx}.npz")
    data = np.load(npzfile, encoding='latin1', allow_pickle=True)
    train_data = data['train']
    for s in range(shot):
        orig_img = train_data[s]
        for m in range(MULTIPLE):
            new_img_path = os.path.join(img_dir, f"{char}-s{s}-{m:02d}.jpg")
            new_img = move_strokes(orig_img)
            show_one_sample(new_img, True, new_img_path)