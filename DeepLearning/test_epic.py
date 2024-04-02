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

shift_interval = -10
prefix = "name"

path = 'datasets/epic'
datasets = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_01.csv"
    mat_file = pd.read_csv(total_path)
    datasets.append(mat_file.iloc[:, [4, 5]])

combined_data = pd.concat(datasets, axis=0)

print(combined_data)

X = combined_data[['hip_sagittal']].values
y = combined_data['hip_sagittal'].shift(shift_interval).fillna(method='ffill').values
X = X.reshape((X.shape[0], 1, X.shape[1]))

model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)
model.save(f"models/{prefix}_last.h5")