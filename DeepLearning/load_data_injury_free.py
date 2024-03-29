import numpy as np
import pandas as pd
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = 'datasets/injury_free/2014001'
os.chdir(path)
datasets = c3d.Reader(open('2014001_C1_01.c3d', 'rb'))
markers_name = datasets.point_labels
other_labels = datasets.analog_labels
print(markers_name)
print(other_labels)
print(f"1: {len(markers_name)}, 2: {len(other_labels)}")
frames = list(datasets.read_frames())
print(frames)
xarr=[]
yarr=[]
zarr=[]
warr=[]
qarr=[]
for x, y, z, w, q in frames[0][1]:
    xarr.append(x)
    yarr.append(y)
    zarr.append(z)
    warr.append(w)
    qarr.append(q)


fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xarr, yarr, zarr)
plt.show()