import pandas as pd
import numpy as np
import os
import configs as config

subjects = config.SUBJECTS
input_window_length = config.INPUT_WINDOW_LENGTH
stride = config.STRIDE

def laplacian_smoothing(data, alpha=0.1, num_iter=3000):
    smoothed_data = data.copy()
    for _ in range(num_iter):
        for i in range(1, len(smoothed_data) - 1): # excluding the first and last points
            smoothed_data[i] = smoothed_data[i] + alpha * (smoothed_data[i-1] + smoothed_data[i+1] - 2 * smoothed_data[i])
    return smoothed_data


folder_path_data_gon = 'EPIC/AB{}/gon'
# folder_path_data_imu = 'EPIC/AB{}/imu'
# folder_path_target = 'EPIC/AB{}/gcRight'

folder_path_smoothed = 'EPIC_smoothed/AB{}/gon'

# Load data
for i in subjects:
    data_path_gon = folder_path_data_gon.format(str(i).zfill(2))
    smoothed_path_gon = folder_path_smoothed.format(str(i).zfill(2))

    if not os.path.exists(smoothed_path_gon):
        os.makedirs(smoothed_path_gon)

    for filename in os.listdir(data_path_gon):
        if filename.endswith('.csv'):
            df_data_gon = pd.read_csv(os.path.join(data_path_gon, filename))
            hip_sagittal = df_data_gon['hip_sagittal'][::5].values
            smoothed = laplacian_smoothing(hip_sagittal)

            save_path = os.path.join(smoothed_path_gon, filename)
            pd.DataFrame(smoothed, columns=['hip_sagittal']).to_csv(save_path, index=False)

