import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
data_name = "levelground_ccw_normal_01_01"
path = f'datasets/epic/AB06/10_09_18/levelground/gon/{data_name}.csv'
data = pd.read_csv(path)
print(data)