import numpy as np 
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
from typing import List
from sklearn import svm

# index of chars. it is so stupid to use Chinese chars in path
def get_index(path):
    idx2chr = os.listdir(path)
    chr2idx = {}
    for i, c in enumerate(idx2chr):
        chr2idx[c] = i
    return idx2chr, chr2idx

idx2chr, chr2idx = get_index('./oracle_fs/img/oracle_200_5_shot/train')

# load data
def load_data(idx2chr: List[str], base_path: str):
    X, y = [], []
    for idx, char in enumerate(idx2chr):
        for image_name in os.listdir(os.path.join(base_path, char)):
            image_path = os.path.join(base_path, char, image_name)
            X.append(np.array(Image.open(image_path)).reshape(-1))
            y.append(idx)
    return np.array(X), np.array(y)

def trainSVM(shot, C=1):
    path = f'./oracle_fs/img/oracle_200_{shot}_shot/'
    X, y = load_data(idx2chr, path+'train')
    X_test, y_test = load_data(idx2chr, path+'test')
    model = svm.SVC(C=C)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    acc = np.sum(y_pred == y_test) / y_pred.size * 100
    return acc, model

acc_5, _ = trainSVM(5, C=8)
print(f"{acc_5:.3f}")  # 45.325

acc_3, _ = trainSVM(3, C=6)
print(f"{acc_3:.3f}")  # 36.000

acc_1, _ = trainSVM(1, C=1)
print(f"{acc_1:.3f}")  # 19.200