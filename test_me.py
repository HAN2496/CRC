import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import configs.config_estimation as config

subjects = config.SUBJECTS
input_window_length = config.INPUT_WINDOW_LENGTH
stride = config.STRIDE

def load_data():
    all_sequences = []
    all_targets = []

    folder_path_data_gon = 'EPIC/AB{}/gon'
    # folder_path_data_imu = 'EPIC/AB{}/imu'
    folder_path_target = 'EPIC/AB{}/gcRight'
    
    # Load data
    for i in subjects:
        data_path_gon = folder_path_data_gon.format(str(i).zfill(2))
        # data_path_imu = folder_path_data_imu.format(str(i).zfill(2))
        target_path = folder_path_target.format(str(i).zfill(2))

        for filename in sorted(os.listdir(data_path_gon)):
            if filename.endswith('.csv'):
                df_data_gon = pd.read_csv(os.path.join(data_path_gon, filename))
                # df_data_imu = pd.read_csv(os.path.join(data_path_imu, filename))
                df_target = pd.read_csv(os.path.join(target_path, filename))
                
                hip_sagittal = df_data_gon['hip_sagittal'][::5].values
                knee_sagittal = df_data_gon['knee_sagittal'][::5].values
                # foot_Accel_X = df_data_imu['foot_Accel_X'].values
                # foot_Accel_Y = df_data_imu['foot_Accel_Y'].values
                # foot_Accel_Z = df_data_imu['foot_Accel_Z'].values
                HeelStrike = df_target['HeelStrike'].values

                heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
                heel_strike_x = np.cos(heel_strike_radians)
                heel_strike_y = np.sin(heel_strike_radians)

                # data = np.column_stack((hip_sagittal, foot_Accel_X, foot_Accel_Y, foot_Accel_Z))
                data = np.column_stack((hip_sagittal, knee_sagittal))
                # data = np.column_stack((hip_sagittal, knee_sagittal, foot_Accel_X, foot_Accel_Y, foot_Accel_Z))
                target = np.column_stack((heel_strike_x, heel_strike_y))
                all_sequences.append(data)
                all_targets.append(target)

    # Split the sequences into training and testing
    train_data, test_data, train_targets, test_targets = train_test_split(all_sequences, all_targets, test_size=0.2, shuffle=False)

    # Apply sliding window to the training data
    X_train, y_train = [], []
    for data, target in zip(train_data, train_targets):
        for start in range(0, len(data) - input_window_length, stride):
            end = start + input_window_length
            X_train.append(data[start:end])
            y_train.append(target[end] if end < len(target) else target[-1])  # Handle edge case for targets

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Number of test sequences: {len(test_data)}, Number of test target sequences: {len(test_targets)}")
    return X_train, y_train, test_data, test_targets
