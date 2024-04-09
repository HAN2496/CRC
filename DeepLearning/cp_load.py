import numpy as np
import pandas as pd
import c3d
import matplotlib.pyplot as plt

path = 'datasets/CPgaitdata-latest/CP child gait data/td'
datasets = []
for i in range(8):
    total_path = f"{path}/"

total_path = 'datasets/CPgaitdata-latest/CP child gait data/td/TD1a.c3d'
datasets = c3d.Reader(open(total_path, 'rb'))
frames = list(datasets.read_frames())
labels = datasets.point_labels
labels2 = datasets.analog_labels
print(frames)
print(labels)
print(labels2)