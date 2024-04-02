import numpy as np
import pandas as pd
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from tsai.all import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import load_model

data_len = 10
print(np.linspace(0, 1, data_len))


shift_interval = -10
prefix = "name"

path = 'datasets/epic'
datasets = []

path='datasets/epic/AB06/10_09_18/levelground/gon/'
data1 = 'levelground_ccw_normal_01_01.csv'
data2 = 'levelground_ccw_normal_01_01.csv'
data1 = pd.read_csv(path+data1)[['hip_sagittal']]
data2 = pd.read_csv(path+data2)[['Header', 'hip_sagittal']]
print(data1)
print(len(data1))
print(len(data2))
data2['Header'] = np.array([i for i in range(len(data2))])
print(data2)
