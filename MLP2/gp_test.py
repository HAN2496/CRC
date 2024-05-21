import numpy as np
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import pandas as pd

#SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
SUBJECTS = [6, 8]

class Datasets:
    def __init__(self):
        self.load_data()
        self.divide_by_section()

    def load_data(self):
        self.data = []
        folder_path_data_gon = 'EPIC/AB{}/gon'
        folder_path_data_imu = 'EPIC/AB{}/imu'
        folder_path_target = 'EPIC/AB{}/gcRight'
        
        # Load data
        for i in SUBJECTS:
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

                    self.data.append({
                        'hip_sagittal': hip_sagittal,
                        'heel_strike_x': heel_strike_x,
                        'heel_strike_y': heel_strike_y,
                        'HeelStrike': HeelStrike
                    })
        self.data = np.array(self.data)
    
    def index_by_scalar(self, start=0, end=100):
        random_test_num = np.random.randint(0, len(self.data))
        entry = self.data[random_test_num]
        start_idx = np.where(entry['HeelStrike'] == start)[0][0]
        end_idx = np.where(entry['HeelStrike'] == end)[0][0]
        selected_data = {
            'hip_angles': entry['hip_sagittal'][start_idx:end_idx+1],
            'heel_strike_x': entry['heel_strike_x'][start_idx:end_idx+1],
            'heel_strike_y': entry['heel_strike_y'][start_idx:end_idx+1],
            'HeelStrike': entry['HeelStrike'][start_idx:end_idx+1]
        }
        return selected_data


    def divide_by_section(self):
        for entry in self.data:
            hip_angles = entry['hip_sagittal']
            heel_strike_indices = self.find_zero_indices(entry['HeelStrike'], 0)

            entry['segments'] = {
                'initial': {
                    'hip_angles': hip_angles[heel_strike_indices[0]:heel_strike_indices[1]],
                    'heel_strike_x': entry['heel_strike_x'][heel_strike_indices[0]:heel_strike_indices[1]],
                    'heel_strike_y': entry['heel_strike_y'][heel_strike_indices[0]:heel_strike_indices[1]],
                    'HeelStrike': entry['HeelStrike'][heel_strike_indices[0]:heel_strike_indices[1]]
                },
                'mid': {
                    'hip_angles': hip_angles[heel_strike_indices[2]:heel_strike_indices[-1]],
                    'heel_strike_x': entry['heel_strike_x'][heel_strike_indices[2]:heel_strike_indices[-1]],
                    'heel_strike_y': entry['heel_strike_y'][heel_strike_indices[2]:heel_strike_indices[-1]],
                    'HeelStrike': entry['HeelStrike'][heel_strike_indices[2]:heel_strike_indices[-1]]
                },
                'final': {
                    'hip_angles': hip_angles[heel_strike_indices[-1]:],
                    'heel_strike_x': entry['heel_strike_x'][heel_strike_indices[-1]:],
                    'heel_strike_y': entry['heel_strike_y'][heel_strike_indices[-1]:],
                    'HeelStrike': entry['HeelStrike'][heel_strike_indices[-1]:]
                }
            }

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return [index for index, value in enumerate(values_list) if value == target_value]

class GP:
    def __init__(self, path="gaussian_process_regressor.joblib"):
        self.path = path
        self.model = load(self.path)
        self.cut_one_cycle()

    def predict(self, X):
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        return y_pred, sigma

    def cut_one_cycle(self):
        indices = np.arange(0, 100, 0.05)
        rad = (indices / 100.0) * 2 * np.pi
        x_rad = np.cos(rad)
        y_rad = np.sin(rad)
        X = np.column_stack((x_rad, y_rad))
        y_pred, sigma = self.predict(X)
        self.X_rad = X
        self.X = indices
        self.y_pred = y_pred
        self.sigma = sigma

    def plot_one_cycle(self):
        plt.plot(self.X, self.y_pred)
        plt.fill_between(self.X, self.y_pred - self.sigma, self.y_pred + self.sigma, color='blue', alpha=0.2,
                        label='Confidence Interval (1 std dev)')
        plt.xlabel('Time')
        plt.ylabel('Hip Sagittal Angle')
        plt.legend()
        plt.show()

start = 30
end = 100
dataset = Datasets()
gp = GP()

selected_data = dataset.index_by_scalar()
print(selected_data)

X = np.column_stack((selected_data['heel_strike_x'], selected_data['heel_strike_y']))
y = selected_data['hip_angles']


y_pred, sigma = gp.predict(X)



plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, 'r.', markersize=10, label='Actual Data (y)')
plt.plot(range(len(y)), y_pred, 'b-', label='Predicted Data (y_pred)')
plt.fill_between(range(len(y)), y_pred - sigma, y_pred + sigma, color='blue', alpha=0.2,
                 label='Confidence Interval (1 std dev)')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time')
plt.ylabel('Hip Sagittal Angle')
plt.legend()
plt.show()











"""


def find_all_indexes(lst, value):
    return [index for index, current in enumerate(lst) if current == value]

hips, heel_rads, heels = load_data()

data_len = len(hips)
print(f"Total data length: {data_len}")

hip_starts, hip_mids, hip_ends = [], [], []
heel_starts, heel_mids, heel_ends = [], [], []
heel_start_rads, heel_mid_rads, heel_end_rads = [], [], []
heel_0_idxs, heel_100_idxs = [], []


for i in range(data_len):
    current_hips = hips[i]
    current_heels = heels[i]
    current_heel_rads = heel_rads[i]
    heel_0_idx = find_all_indexes(heels[i], 0)
    heel_0_idxs.append(heel_0_idx)
    heel_100_idx = find_all_indexes(heels[i], 100)
    heel_100_idxs.append(heel_100_idx)

    tmp2 = 3
    hip_starts.append(current_hips[heel_0_idx[0]:heel_0_idx[tmp2]])
    heel_starts.append(current_heels[heel_0_idx[0]:heel_0_idx[tmp2]])

    tmp = -3
    hip_mids.append(current_hips[heel_0_idx[tmp2]:heel_100_idx[tmp]])
    heel_mids.append(current_heels[heel_0_idx[tmp2]:heel_100_idx[tmp]])
    heel_mid_rads.append(current_heel_rads[heel_0_idx[tmp2]:heel_100_idx[tmp], :])

    hip_ends.append(current_hips[heel_100_idx[tmp]+1:heel_100_idx[-1]])
    heel_ends.append(current_heels[heel_100_idx[tmp]+1:heel_100_idx[-1]])


#X = heel_mids[0].reshape(-1, 1)
random_test_num = np.random.randint(0, data_len)
sliding_pos_i, sliding_pos_f = 300, 600
X = heel_mid_rads[random_test_num][sliding_pos_i:sliding_pos_f]
y = hip_mids[random_test_num]
print(f"x shape: {X.shape}, y shape: {y.shape}")


gp = load("gaussian_process_regressor.joblib")
y_pred, sigma = gp.predict(X, return_std=True)

plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, 'r.', markersize=10, label='Actual Data (y)')
plt.plot(range(sliding_pos_i, sliding_pos_f), y_pred, 'b-', label='Predicted Data (y_pred)')
plt.fill_between(range(sliding_pos_i, sliding_pos_f), y_pred - sigma, y_pred + sigma, color='blue', alpha=0.2,
                 label='Confidence Interval (1 std dev)')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time')
plt.ylabel('Hip Sagittal Angle')
plt.legend()
plt.show()
"""