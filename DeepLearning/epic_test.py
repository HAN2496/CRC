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

prefix = 'name'
model = load_model(f'models/{prefix}_last.h5')
print(model.summary())