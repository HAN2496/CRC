import numpy as np
import pandas as pd
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from tsai.all import *
import numpy as np
from tsai.all import *
from sklearn.model_selection import train_test_split

shift_interval = -10

path = 'datasets/epic'
datasets = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_01.csv"
    mat_file = pd.read_csv(total_path.iloc[:, [0, 4]])

    datasets.append(mat_file)
    header_len = len()

combined_data = pd.concat(datasets, axis=0)

print(combined_data)



X = np.array([df.to_numpy() for df in combined_data])
y = np.arange(len(combined_data))  
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

splits = (list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_valid))))
tfms = [None, [Categorize()]]
dls = get_ts_dls(X_train, y_train, X_valid, y_valid, tfms=tfms, splits=splits)

model = LSTM(dls.vars, dls.c, hidden_size=100, num_layers=1, bidirectional=False)
learn = Learner(dls, model, metrics=accuracy)
learn.fit_one_cycle(10, 1e-3)