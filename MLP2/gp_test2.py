import os
import time
import numpy as np
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import minimize

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
    def __init__(self, data_num, path='gaussian_process_regressor.joblib'):
        self.data_num = data_num
        self.path = path
        self.model = load(self.path)
        self.predict_by_scalar(data_num)
        self.X_original = self.X.copy()
        self.X_scalar_original = self.X_scalar.copy()
        self.y_pred_original = self.y_pred.copy()

    def predict(self, X): #polar coordinate으로 predict 뽑는게 디폴트
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        self.X = X
        self.y_pred = y_pred
        self.sigma = sigma

        return y_pred, sigma

    def predict_by_scalar(self, num, start=0, end=100): #scalar 값으로 predict 계산시 사용
        HeelStrike = np.linspace(start, end, num)
        self.X_scalar = HeelStrike
        heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
        heel_strike_x = np.cos(heel_strike_radians)
        heel_strike_y = np.sin(heel_strike_radians)
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return self.predict(X)

    def find_pred_value_by_scalar(self, x_scalar): # (HeelStrike) Scalar 값으로 gp 모델의 predict 값 반환
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

    def scale(self, scale_x=1, scale_y=1, x_pos=0, y_pos=0, save_data=False):
        X_scalar_gp = self.X_scalar.copy() - x_pos
        y_pred_gp = self.y_pred.copy() - y_pos

        X_scalar_gp *= scale_x
        y_pred_gp *= scale_y

        X_scalar_gp += x_pos
        y_pred_gp += y_pos

        if save_data == False:
            return X_scalar_gp, y_pred_gp
        else:
            self.X_scalar = X_scalar_gp
            self.y_pred = y_pred_gp

class Control:
    def __init__(self):
        self._init_datasets()
        self._init_gp()
    
    def _init_datasets(self):
        self.dataset = Datasets()
        self.collect_start, self.collect_end = 0, 100
        self.selected_data = self.dataset.index_by_scalar(start=self.collect_start, end=self.collect_end)
        self.X = np.column_stack((self.selected_data['heel_strike_x'], self.selected_data['heel_strike_y']))
        self.X_scalar = self.selected_data['heel_strike']
        self.y = self.selected_data['hip_sagittal']
        self.X_scalar_end, self.y_end = self.X_scalar[-1], self.y[-1]
        self.data_num = len(self.y)
    
    def _init_gp(self):
        self.interval = 20
        self.predict_start = self.collect_end - self.interval
        self.predict_end = self.collect_end + self.interval

        # 모델 정의
        self.gp = GP(data_num=self.data_num)

        y_pred_target = self.gp.find_pred_value_by_scalar(self.X_scalar_end)
        self.gp.translation(delx=0, dely=y_pred_target - self.y_end) # 테스트 데이터셋의 끝점과 맞춰주기 위해 평행이동

    def calculate_contour_error(self, X_actual, y_actual, X_pred, y_pred, num_points=500):
        if self.predict_start >= self.collect_end:
            return 0
        x_common = np.linspace(self.predict_start + 0.1, self.collect_end - 0.1, num_points)
        
        interpolator_actual = interp1d(X_actual, y_actual, kind='linear')
        y_actual_interp = interpolator_actual(x_common) # actual 값들 공통 x지점으로

        interpolator_pred = interp1d(X_pred, y_pred, kind='linear')
        y_pred_interp = interpolator_pred(x_common)

        mse = np.mean(np.square((y_actual_interp - y_pred_interp)))
        return mse

    def gradient_descent_algorithm(self, initial_scale_x, initial_scale_y, learning_rate, num_iterations, data):
        print_result = 0
        epsilon = 0.005
        scale_x = initial_scale_x
        scale_y = initial_scale_y
        X_actual, y_actual = data
        scales_history = []

        for i in range(num_iterations):
            X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x, scale_y, self.X_scalar_end, self.y_end)
            contour_error = self.calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
            scales_history.append((scale_x, scale_y, contour_error))

            if print_result % 50 == 0:
                print(f"Iteration {i}: Contour Error = {contour_error}, Scale_X = {scale_x}, Scale_Y = {scale_y}")
        
            X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x + epsilon, scale_y, self.X_scalar_end, self.y_end)
            error_x_plus = self.calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
            X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x - epsilon, scale_y, self.X_scalar_end, self.y_end)
            error_x_minus = self.calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
            gradient_x = (error_x_plus - error_x_minus) / (2 * epsilon)

            X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x, scale_y + epsilon, self.X_scalar_end, self.y_end)
            error_y_plus = self.calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
            X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x, scale_y - epsilon, self.X_scalar_end, self.y_end)
            error_y_minus = self.calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
            gradient_y = (error_y_plus - error_y_minus) / (2 * epsilon)

            scale_x -= learning_rate * gradient_x
            scale_y -= learning_rate * gradient_y
            print_result +=1

        return scale_x, scale_y, scales_history

    def update_frame(self, i, scales_history, X_actual, y_actual, line1, line2, title):
        scale_x, scale_y, _ = scales_history[i]
        X_scalar_pred_new, y_pred_new = self.gp.scale(scale_x, scale_y, self.X_scalar_end, self.y_end)
        line1.set_data(X_actual, y_actual)
        line2.set_data(X_scalar_pred_new, y_pred_new)
        title.set_text(f'Iteration {i}: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}')
        return line1, line2

    def visualize(self, scales_history, frames_interval=10, fps=60):
        fig, ax = plt.subplots()
        line1, = ax.plot([], [], 'r.', markersize=10, label='Actual Data')
        line2, = ax.plot([], [], 'b-', label='Predicted Data')
        title = ax.set_title('')
        ax.legend()
        ax.set_xlim(min(self.X_scalar), max(self.X_scalar) + 20)
        ax.set_ylim(min(self.y) - 10, max(self.y) + 10)

        frames = range(0, len(scales_history), frames_interval)
        ani = animation.FuncAnimation(fig, self.update_frame, frames=frames, interval=1, fargs=(scales_history, self.X_scalar, self.y, line1, line2, title))

        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save('contour_error_animation.mp4', writer=writervideo)

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(X_scalar, y, 'r.', markersize=10, label='Actual Data (y)')
        plt.plot(gp.X_scalar_original, gp.y_pred_original, 'b-', label="Predicted Data (initial)")
        plt.plot(gp.X_scalar, gp.y_pred, 'k-', label='After move gp to end point')
        gp_testx, gp_testy = gp.scale(2.0, 2.0, X_scalar_end, y_end)
        plt.plot(gp_testx, gp_testy, 'g-', label='size up (twice)')
        gp_testx, gp_testy = gp.scale(0.5, 0.5, X_scalar_end, y_end)
        plt.plot(gp_testx, gp_testy, 'y-', label='size down (half)')
        plt.title('Check translation')
        plt.xlabel('Time')
        plt.ylabel('Hip Sagittal Angle')
        plt.legend()
        plt.show()


plot_process = False
"""
Step 0. 테스트 데이터셋 구축. 0 % ~ 41 % 사이의 데이터가 들어왔다고 가정
"""
dataset = Datasets()
collect_start = 0 # 맨 처음값부터 사용
collect_end = 50
selected_data = dataset.index_by_scalar(start=collect_start, end=collect_end)
X = np.column_stack((selected_data['heel_strike_x'], selected_data['heel_strike_y']))
X_scalar =  selected_data['heel_strike']
y = selected_data['hip_sagittal']
X_scalar_end, y_end = X_scalar[-1], y[-1]
data_num = len(y)

"""
Step 1. gp 파트. 끝 지점을 기준으로 -20 % ~ +20 % 지점을 정의해줌.
"""
interval = 45
predict_start = collect_end - interval
predict_end = collect_end + interval

#모델 정의
gp = GP(data_num=data_num)

y_pred_target = gp.find_pred_value_by_scalar(X_scalar_end)
gp.translation(delx=0, dely=y_pred_target - y_end) # 테스트 데이터셋의 끝점과 맞춰주기 위해 평행이동

if plot_process:
    plt.figure(figsize=(10, 6))
    plt.plot(X_scalar, y, 'r.', markersize=10, label='Actual Data (y)')
    plt.plot(gp.X_scalar_original, gp.y_pred_original, 'b-', label="Predicted Data (initial)")
    plt.plot(gp.X_scalar, gp.y_pred, 'k-', label='After move gp to end point')
    #gp_testx, gp_testy = gp.scale(2.0, 2.0, X_scalar_end, y_end)
    #plt.plot(gp_testx, gp_testy, 'g-', label='size up (twice)')
    #gp_testx, gp_testy = gp.scale(0.5, 0.5, X_scalar_end, y_end)
    #plt.plot(gp_testx, gp_testy, 'y-', label='size down (half)')
    plt.title('Check translation')
    plt.xlabel('Time')
    plt.ylabel('Hip Sagittal Angle')
    plt.legend()
    plt.show()


#countouring 계산 및 plot 용 데이터 (0 % ~ 61 % 지점)
compare_data = dataset.index_by_scalar(start=collect_start, end=predict_end)
X_compare = np.column_stack((compare_data['heel_strike_x'], compare_data['heel_strike_y']))
X_scalar_compare = compare_data['heel_strike']
y_compare = compare_data['hip_sagittal']
header_compare = np.linspace(predict_start, predict_end, int(data_num * (predict_end - predict_start) / (predict_end - collect_start)))

#중간 확인
if plot_process:
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

"""
Stpe2: contour error 계산
 predict_start ~ collect_end 까지의 contour error를 계산해야함.
 gp의 predict는 상황에 따라 바뀔 수 있어서 함수의 인자로 포함되어 있어야함.
 문제점: heel strike scalar 값의 증가폭이 동일하지 않음.
 해결방안: 선형보간으로 x를 동일하게 만든 뒤 계산하였음.
"""

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, num_points=500):
    if predict_start >= collect_end:
        return 0
    x_common = np.linspace(predict_start+0.1, collect_end-0.1, num_points)
    
    interpolator_actual = interp1d(X_actual, y_actual, kind='linear')
    y_actual_interp = interpolator_actual(x_common) #actual 값들 공통 x지점으로

    interpolator_pred = interp1d(X_pred, y_pred, kind='linear')
    y_pred_interp = interpolator_pred(x_common)

    mse = np.mean(np.square((y_actual_interp - y_pred_interp)))
    return mse

if plot_process:
    contour_error = calculate_contour_error(X_scalar_compare, y_compare, gp.X_scalar, gp.y_pred)
    print(f"Contour Error: {contour_error}")

"""
Step3: gradient descent algorithm 적용
scipy.optimize.minimize
"""

def gradient_descent_algorithm(initial_scale_x, initial_scale_y, learning_rate, num_iterations, data):
    print_result = 0
    epsilon = 0.005
    scale_x = initial_scale_x
    scale_y = initial_scale_y
    X_actual, y_actual = data
    scales_history = []

    for i in range(num_iterations):
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y, X_scalar_end, y_end)
        contour_error = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
        scales_history.append((scale_x, scale_y, contour_error))

        if print_result % 50 == 0:
            print(f"Iteration {i}: Contour Error = {contour_error}, Scale_X = {scale_x}, Scale_Y = {scale_y}")
    
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x + epsilon, scale_y, X_scalar_end, y_end)
        error_x_plus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x - epsilon, scale_y, X_scalar_end, y_end)
        error_x_minus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
        gradient_x = (error_x_plus - error_x_minus) / (2 * epsilon)

        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y + epsilon, X_scalar_end, y_end)
        error_y_plus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y - epsilon, X_scalar_end, y_end)
        error_y_minus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new)
        gradient_y = (error_y_plus - error_y_minus) / (2 * epsilon)

        scale_x -= learning_rate * gradient_x
        scale_y -= learning_rate * gradient_y
        print_result +=1

    return scale_x, scale_y, scales_history

data = (X_scalar, y)
#data = (X_scalar_compare, y_compare)
tic = time.time()
final_scale_x, final_scale_y, scales_history = gradient_descent_algorithm(1.0, 1.0, 0.001, 500, data)
toc = time.time()
print("inference time:", toc - tic)

"""
Step 4: 시각화
"""
def update_frame(i, scales_history, X_actual, y_actual, line1, line2, line3, title):
    scale_x, scale_y, _ = scales_history[i]
    X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y, X_scalar_end, y_end)
    X_scalar_pred_new2, y_pred_new2 = gp.scale(1.0, 1.0, X_scalar_end, y_end)
    line1.set_data(X_actual, y_actual)
    line2.set_data(X_scalar_pred_new, y_pred_new)
    line3.set_data(X_scalar_pred_new2, y_pred_new2)
    title.set_text(f'Iteration {i}: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}')
    return line1, line2

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'r.', markersize=10, label='Actual Data')
line2, = ax.plot([], [], 'b-', label='Predicted Data')
line3, = ax.plot([], [], 'g-', label='Initial GP Data')
title = ax.set_title('')
ax.legend()
ax.set_xlim(min(X_scalar), 100)
ax.set_ylim(min(y) - 10, max(y) + 10)

#interval이 속도 결정하는듯. 작을수록 빠름
frames=range(0, len(scales_history), 1) #<- 이렇게 하면 2개씩만
ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=2, fargs=(scales_history, X_scalar, y, line1, line2, line3, title))

writervideo = animation.FFMpegWriter(fps=60)

ani.save('gradient_descent_plot.gif', writer='imagemagick', fps=30, dpi=100)
plt.show()

"""
def optimization_target(scale_params, X_actual, y_actual, X_pred_base, y_pred_base):
    scale_x, scale_y = scale_params
    scaled_X_pred = X_pred_base * scale_x
    scaled_y_pred = y_pred_base * scale_y

    # 실제 데이터와 예측 데이터에 대해 x축의 값이 서로 다를 경우, 보간을 수행
    common_x = np.linspace(min(X_actual), max(X_actual), len(X_actual))  # 보간할 공통 x축 값 생성
    interp_actual = interp1d(X_actual, y_actual, kind='linear', fill_value="extrapolate")
    interp_pred = interp1d(scaled_X_pred, scaled_y_pred, kind='linear', fill_value="extrapolate")

    y_actual_interp = interp_actual(common_x)
    y_pred_interp = interp_pred(common_x)

    # 계산된 y값들에 대한 MSE 계산
    mse = np.mean((y_actual_interp - y_pred_interp) ** 2)
    return mse



initial_guess = [1, 1]  # 초기 스케일링 인자
bounds = [(0.1, 2), (0.1, 2)]  # 스케일 인자에 대한 경계

# 최적화 실행
result = minimize(optimization_target, initial_guess, args=(X, y, X_pred_base, y_pred_base), bounds=bounds, method='L-BFGS-B')

# 최적화 결과 출력
print("Optimal scaling factors:", result.x)
print("Minimum MSE:", result.fun)
"""