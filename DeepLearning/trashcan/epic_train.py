import numpy as np
import pandas as pd
import tensorflow as tf
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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


shift_interval = +10
prefix = "name"

path = 'datasets/epic'
datasets = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    if i==0:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_02_01.csv"
    else:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_02.csv"
    mat_file = pd.read_csv(total_path).iloc[:, [0, 4]]
    data_len = len(mat_file)
    mat_file['Header'] = np.linspace(0, 1, data_len)
    datasets.append(mat_file)

    if i == 1:
        print("*"*50)
        print("Check data")
        print(mat_file)
        print("*"*50)

combined_data = pd.concat(datasets, axis=0)

print(combined_data)





X = combined_data[['Header', 'hip_sagittal']].values
y = combined_data[['Header', 'hip_sagittal']].shift(shift_interval).fillna(method='ffill').values
X = X.reshape((X.shape[0], len(X), X.shape[1]))

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)


model_save_path = f"models/{prefix}_last.h5"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)