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

def load_data(remove_rot=None, remove_vel=None):
    X, y = [], []
    heel_strike = []
    folder_path_data_gon = 'EPIC/AB{}/gon'
    folder_path_data_imu = 'EPIC/AB{}/imu'
    folder_path_target = 'EPIC/AB{}/gcRight'
    
    # Load data
    for i in subjects:
        data_path_gon = folder_path_data_gon.format(str(i).zfill(2))
        # data_path_imu = folder_path_data_imu.format(str(i).zfill(2))
        target_path = folder_path_target.format(str(i).zfill(2))

        for filename in os.listdir(data_path_gon):
            if filename.endswith('.csv') and "ccw_normal" in filename:
                df_data_gon = pd.read_csv(os.path.join(data_path_gon, filename))
                # df_data_imu = pd.read_csv(os.path.join(data_path_imu, filename))
                df_target = pd.read_csv(os.path.join(target_path, filename))
                
                hip_sagittal = df_data_gon['hip_sagittal'][::5].values
                # foot_Accel_X = df_data_imu['foot_Accel_X'].values
                # foot_Accel_Y = df_data_imu['foot_Accel_Y'].values
                # foot_Accel_Z = df_data_imu['foot_Accel_Z'].values
                HeelStrike = df_target['HeelStrike'].values

                heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
                heel_strike_x = np.cos(heel_strike_radians)
                heel_strike_y = np.sin(heel_strike_radians)

                X.append(hip_sagittal)
                y.append(np.column_stack((heel_strike_x, heel_strike_y)))
                heel_strike.append(HeelStrike)

    return X, y, heel_strike

if __name__ == "__main__":
    X, y, _ = load_data()
    testx = np.array(X[0])
    testy = np.array(y[0])
    print(testx.shape, testy.shape)