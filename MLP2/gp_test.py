import numpy as np
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import math
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

from scipy.interpolate import interp1d

#SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
SUBJECTS = [6, 8]

def polar_scalar(x, y):
    angle_rad = math.atan2(y, x)    
    if angle_rad < 0:
        angle_rad += 2 * math.pi    
    scalar_value = angle_rad / (2 * math.pi) * 100    
    return scalar_value


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
    def __init__(self, path='gaussian_process_regressor.joblib'):
        self.path = path
        self.model = load(self.path)
    
    def predict(self, X):
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        self.X = X
        self.y_pred = y_pred
        self.sigma = sigma
        return y_pred, sigma

    def predict_by_range(self, num, start=0, end=100):
        HeelStrike = np.linspace(start, end, num)
        self.X_scalar = HeelStrike
        heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
        heel_strike_x = np.cos(heel_strike_radians)
        heel_strike_y = np.sin(heel_strike_radians)
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return self.predict(X)

    def find_pred_value_by_heelstrike_scalar(self, x_scalar):
        if x_scalar in self.X_scalar:
            return self.y_pred[np.where(x_scalar == self.X_scalar)]
        else:
            closest_indices = np.argsort(np.abs(self.X_scalar - x_scalar))[:2]
            closest_x_values = self.X_scalar[closest_indices]
            closest_y_values = self.y_pred[closest_indices]
            x0, x1 = closest_x_values
            y0, y1 = closest_y_values
            interpolated_value = y0 + (y1 - y0) * (x_scalar - x0) / (x1 - x0)
            return interpolated_value

    def translation(self, delx=0, dely=0):
        self.X_scalar += delx
        self.y_pred -= dely

    def scale(self, scale_x=1, scale_y=1, x_pos=0, y_pos=0):
        self.X_scalar -= x_pos
        self.y_pred -= y_pos

        self.X_scalar *= scale_x
        self.y_pred *= scale_y

        self.X_scalar += x_pos
        self.y_pred += y_pos

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
y_pred_target = gp.find_pred_value_by_heelstrike_scalar(X_scalar[-1])
gp.translation(delx=0, dely=y_pred_target - y[-1])
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
plt.plot(gp.X_scalar, gp.y_pred, 'b-', label='Predicted Data (y_pred)')
plt.fill_between(gp.X_scalar, gp.y_pred - gp.sigma, gp.y_pred + gp.sigma, color='blue', alpha=0.2,
                 label='Confidence Interval (1 std dev)')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time')
plt.ylabel('Hip Sagittal Angle')
plt.legend()
plt.show()



from scipy.interpolate import interp1d
from scipy.integrate import simps

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, interval_end, num_points=500):
    interval_start = min(X_pred)
    
    if interval_start >= interval_end:
        return 0  # No valid range to calculate error over
    x_common = np.linspace(interval_start, interval_end, num_points)
    
    interpolator_actual = interp1d(X_actual, y_actual, kind='linear', fill_value='extrapolate')
    y_actual_interpolated = interpolator_actual(x_common)

    interpolator_pred = interp1d(X_pred, y_pred, kind='linear', fill_value='extrapolate')
    y_pred_interpolated = interpolator_pred(x_common)
    
    absolute_difference = np.abs(y_actual_interpolated - y_pred_interpolated)
    contour_error = simps(absolute_difference, x_common)
    normalized_contour_error = contour_error / (interval_end - interval_start) * 10
    
    return normalized_contour_error

interval_start = test_end - interval
interval_end = test_end
contour_error = calculate_contour_error(X_scalar_compare, y_compare, gp.X_scalar, gp.y_pred, interval_end)
print(f"Contour Error: {contour_error}")



X_gp_before, y_gp_before = gp.X_scalar.copy(), gp.y_pred.copy()
plt.plot(X_gp_before, y_gp_before, 'b-', label='Predicted Data (y_pred)')
plt.plot(gp.X_scalar, gp.y_pred, 'g-', label='Predicted Data (y_pred)')
plt.show()

def gradient_descent_scale(gp, initial_scale_x, initial_scale_y, learning_rate, num_iterations, X_actual, y_actual, interval_start, interval_end):
    scale_x = initial_scale_x
    scale_y = initial_scale_y
    
    for iteration in range(num_iterations):
        gp.scale(scale_x, scale_y, X_actual[-1], y_actual[-1])
      
        current_error = calculate_contour_error(X_actual, y_actual, gp.X_scalar, gp.y_pred, interval_end)
        
        #print(f"Iteration {iteration+1}, Scale X: {scale_x}, Scale Y: {scale_y}, Error: {current_error}")
        
        derv = 1.0 * (1 + 0.01) # 1% 만큼 변화를 줌
        derv_inv = 1.0 / (1 + 0.01)
        gp.scale(derv, 1.0, X_actual[-1], y_actual[-1])  # Small perturbation in X scale
        error_x_perturb = calculate_contour_error(X_actual, y_actual, gp.X_scalar, gp.y_pred, interval_end)

        gp.scale(derv_inv, 1.0, X_actual[-1], y_actual[-1])  # Reset X scale perturbation and apply Y
        gp.scale(1.0, derv, X_actual[-1], y_actual[-1])  # Small perturbation in Y scale
        error_y_perturb = calculate_contour_error(X_actual, y_actual, gp.X_scalar, gp.y_pred, interval_end)

        grad_x = (error_x_perturb - current_error) / (0.01 * scale_x)
        grad_y = (error_y_perturb - current_error) / (0.01 * scale_y)
        #print(f"grad x: {grad_x}, grad y: {grad_y}")
        
        # Update scales
        scale_x -= learning_rate * grad_x
        scale_y -= learning_rate * grad_y
        plt.plot(gp.X_scalar, gp.y_pred, 'g-', label='Predicted Data (y_pred)')
        plt.show()
    return scale_x, scale_y

# Parameters
initial_scale_x = 1.0
initial_scale_y = 1.0
learning_rate = 0.0001
num_iterations = 500

# Assuming `gp`, `X_scalar_compare`, and `y_compare` are defined from previous sections
optimized_scale_x, optimized_scale_y = gradient_descent_scale(
    gp, initial_scale_x, initial_scale_y, learning_rate, num_iterations, 
    X_scalar, y, interval_start, interval_end)

print(f"Optimized Scale X: {optimized_scale_x}, Optimized Scale Y: {optimized_scale_y}")


print(X_scalar[-1], y[-1])
gp.scale(optimized_scale_x, optimized_scale_y, X_scalar[-1], y[-1])

#plt.plot(X_scalar, y, 'r.', markersize=10, label='Actual Data (y)')
#plt.plot(X_gp_before, y_gp_before, 'b-', label='Predicted Data (y_pred)')
plt.plot(gp.X_scalar, gp.y_pred, 'g-', label='Predicted Data (y_pred)')
plt.show()