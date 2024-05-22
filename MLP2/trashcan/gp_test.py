import numpy as np
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel


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
                    header = df_target['Header'].values
                    heel_strike_speed = np.diff(HeelStrike, prepend=0)

                    heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
                    heel_strike_x = np.cos(heel_strike_radians)
                    heel_strike_y = np.sin(heel_strike_radians)

                    self.data.append({
                        'header': header,
                        'hip_sagittal': hip_sagittal,
                        'heel_strike_x': heel_strike_x,
                        'heel_strike_y': heel_strike_y,
                        'heel_strike': HeelStrike,
                        'heel_strike_speed': heel_strike_speed
                    })
        self.data = np.array(self.data)

    def divide_by_section(self):
        self.heel_strike_indices = []
        for entry in self.data:
            heel_strike_indices = self.find_zero_indices(entry['heel_strike'], 0)
            self.heel_strike_indices.append(heel_strike_indices)

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return [index for index, value in enumerate(values_list) if value == target_value]

    def index_by_scalar(self, start=0, end=90, index_pos=4):
        start, end = int(start), int(end)
        random_test_num = np.random.randint(0, len(self.data))
        random_test_num = 0
        entry = self.data[random_test_num]
        start_idx = self.heel_strike_indices[random_test_num][4]
        end_idx = self.heel_strike_indices[random_test_num][5] - 1

        heel_strike_indices = self.heel_strike_indices[random_test_num]

        if start < 0:
            num, val = index_pos-1, 100 + start
        else:
            num, val = index_pos, start

        start_idx_within_stride = np.argmin(np.abs(entry['heel_strike'][heel_strike_indices[num]:heel_strike_indices[num+1]] - val))
        start_idx = heel_strike_indices[num] + start_idx_within_stride

        end_idx_within_stride = np.argmin(np.abs(entry['heel_strike'][heel_strike_indices[4]:heel_strike_indices[5]] - end))
        end_idx = heel_strike_indices[4] + end_idx_within_stride

        selected_data = {
            'hip_sagittal': entry['hip_sagittal'][start_idx:end_idx+1],
            'heel_strike_x': entry['heel_strike_x'][start_idx:end_idx+1],
            'heel_strike_y': entry['heel_strike_y'][start_idx:end_idx+1],
            'heel_strike': entry['heel_strike'][start_idx:end_idx+1]
        }
        return selected_data

class GP:
    def __init__(self, path=None):
        self.path = path
        if path is not None:
            self.model = load(self.path)
        else:
            print("Should enter path or should fit")

    def fit(self, X, y):
        kernel = ExpSineSquared(length_scale=3.0, periodicity=3.0,
                                length_scale_bounds=(0.1, 100.0),
                                periodicity_bounds=(0.1, 100.0)
                                ) + WhiteKernel(noise_level=0.1)

        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)
        dump(model, 'gp_model.joblib')
        self.model = model
    
    def predict(self, X):
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        self.X = X
        return y_pred, sigma



class GP:
    def __init__(self, path="gaussian_process_regressor.joblib"):
        self.path = path
        self.model = load(self.path)
        self.cut_one_cycle()

    def predict(self, X):
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        self.X = X
        return y_pred, sigma

    def make_heel_strike_val(self, num, start=0, end=100):
        HeelStrike = np.linspace(start, end, num)
        self.X_scalar = HeelStrike
        heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
        heel_strike_x = np.cos(heel_strike_radians)
        heel_strike_y = np.sin(heel_strike_radians)
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return X

    def predict_by_range(self, num, start=0, end=100):
        X = self.make_heel_strike_val(num, start, end)
        return self.predict(X)

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

    def plot(self, start=0, end=100):
        X = self.make_heel_strike_val(start, end)
        y_pred = self.predict_by_range(start, end)
        plt.plot(self.X, self.y_pred)
        plt.fill_between(self.X, self.y_pred - self.sigma, self.y_pred + self.sigma, color='blue', alpha=0.2,
                        label='Confidence Interval (1 std dev)')
        plt.xlabel('Time')
        plt.ylabel('Hip Sagittal Angle')
        plt.legend()
        plt.show()

dataset = Datasets()
gp = GP()

class Control:
    def __init__(self):
        self.dataset = Datasets()
        self.gp = GP()
    
    def get_test_datasets(self, start, end):
        selected_data = dataset.index_by_scalar(start=start, end=end)
        X = np.column_stack((selected_data['heel_strike_x'], selected_data['heel_strike_y']))
        y = selected_data['hip_sagittal']
        return X, y

    def calc_contour_error(self):
        pass

    def fit(self):
        pass

"""
Step 0. 테스트 데이터셋 구축. 0 % ~ 41 % 사이의 데이터가 들어왔다고 가정
"""
test_start = 0
test_end = 41
selected_data = dataset.index_by_scalar(start=test_start, end=test_end)
X = np.column_stack((selected_data['heel_strike_x'], selected_data['heel_strike_y']))
X_scalar =  selected_data['heel_strike']
y = selected_data['hip_sagittal']
data_num = len(y)
header = np.linspace(test_start, test_end, data_num)

"""
Step 1. gp 파트. 끝 지점을 기준으로 -20 ~ 0 지점의 countour error를 계산
"""
interval = 20
start = test_end - interval
end = test_end + interval

y_pred, sigma = gp.predict_by_range(data_num, start, end)
header_gp = np.linspace(start, end, int(data_num * (end - start) / (test_end - test_start)))

#countouring 계산 및 plot 용 데이터
compare_data = dataset.index_by_scalar(start=test_start, end=end)
X_compare = np.column_stack((compare_data['heel_strike_x'], compare_data['heel_strike_y']))
X_scalar_compare = compare_data['heel_strike']
y_compare = compare_data['hip_sagittal']
header_compare = np.linspace(test_start, end, int(data_num * (end - start) / (end - test_start)))

#중간 확인
plt.figure(figsize=(10, 6))
plt.plot(X_scalar, y, 'r.', markersize=10, label='Actual Data (y)')
plt.plot(gp.X_scalar, y_pred, 'b-', label='Predicted Data (y_pred)')
plt.fill_between(gp.X_scalar, y_pred - sigma, y_pred + sigma, color='blue', alpha=0.2,
                 label='Confidence Interval (1 std dev)')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time')
plt.ylabel('Hip Sagittal Angle')
plt.legend()
plt.show()

