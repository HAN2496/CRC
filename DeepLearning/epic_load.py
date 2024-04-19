import numpy as np
import pandas as pd
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
path = 'datasets/epic'

check_len = []

datasets = []
data_num = [i for i in range(25) if i!=16 and i!=20 and i!=23]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_01.csv"
    mat_file = pd.read_csv(total_path)
    datasets.append(mat_file)
    check_len.append(len(mat_file.iloc[:, 0].values))




path = 'datasets/epic'
patient_path = f"{path}/AB{str(6).zfill(2)}"
folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"][0]

gon_datasets = []
imu_datasets = []
phase_datasets = []

gon_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_02_01.csv"
imu_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_02_01.csv"
phase_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_ccw_normal_02_01.csv"
gon_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_02_01.csv"
phase_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_cw_normal_02_01.csv"

gon_file = pd.read_csv(gon_path_ccw).iloc[::5, 4].values
imu_file = pd.read_csv(imu_path_ccw).iloc[:, 4].values
phase_file = pd.read_csv(phase_path_ccw).iloc[:, 1].values


zero_indices = np.where(phase_file == 0)[0]
hundred_indices = np.where(phase_file == 100)[0]

print("Indices where the value is 0 in phase_file:", zero_indices)
print("Indices where the value is 100 in phase_file:", hundred_indices)


gon_file = gon_file[:]
phase_file = phase_file[:]


fig = plt.figure(figsize=(20,12)) ## 캔버스 생성fig.set_facecolor('white')
ax1 = fig.add_subplot()
ax1.plot([i for i in range(len(gon_file))], gon_file, color='blue', label='gon')
ax1.set_xlabel('time step (200 Hz)', fontsize=20)
ax1.set_ylabel('Goniometer', color='blue', fontsize=20)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=20)
ax1.set_title("Goniometer values and Heelstrike in total datasets", fontsize=30)

#plt.plot([i for i in range(len(imu_file))], imu_file, color='blue', label='imu')

ax2 = ax1.twinx()
ax2.plot([i for i in range(len(phase_file))], phase_file, color='red', label='HeelStrike')
ax2.set_ylabel('phase', color='red', fontsize=20)
ax2.tick_params(axis='y', labelcolor='red', labelsize=20)
plt.show()

"""
plt.plot(phase_file, gon_file)
plt.show()
"""