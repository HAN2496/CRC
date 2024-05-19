import numpy as np
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump, load
"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# 커널 클래스 정의
class ExpSineSquaredKernel:
    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
        sin_dists = np.sin(np.pi * np.sqrt(dists) / self.periodicity) ** 2
        return np.exp(-2 * sin_dists / self.length_scale ** 2)

    def get_params(self):
        return {
            'length_scale': self.length_scale,
            'periodicity': self.periodicity
        }

class WhiteKernel:
    def __init__(self, noise_level=1.0):
        self.noise_level = noise_level

    def __call__(self, X, Y=None):
        if Y is None:
            return self.noise_level * np.eye(X.shape[0])
        elif X.shape == Y.shape:
            return self.noise_level * np.eye(X.shape[0])
        else:
            return np.zeros((X.shape[0], Y.shape[0]))

    def get_params(self):
        return {
            'noise_level': self.noise_level
        }

class SumKernel:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        K1 = self.k1(X, Y)
        K2 = self.k2(X, Y)
        return K1 + K2

    def get_params(self):
        params = self.k1.get_params()
        params.update(self.k2.get_params())
        return params

# GaussianProcessRegressorCustom 클래스 정의
class GaussianProcessRegressorCustom:
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        K = self.kernel(X) + 1e-10 * np.eye(len(X))  # Add small noise for numerical stability
        self.K_inv = np.linalg.inv(K)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        K_trans = self.kernel(self.X_train, X)
        K_pred = self.kernel(X)
        y_mean = K_trans.T @ self.K_inv @ self.y_train
        y_var = K_pred - K_trans.T @ self.K_inv @ K_trans
        return y_mean, np.sqrt(np.diag(y_var))

# 학습된 Gaussian Process Regressor 로드 및 매개변수 출력
gp = load('gaussian_process_regressor.joblib')
print("Learned kernel parameters:")
kernel_1 = gp.kernel_.k1  # ExpSineSquared 커널
kernel_2 = gp.kernel_.k2  # WhiteKernel 커널

print(f"ExpSineSquared kernel parameters:\n{kernel_1}\n")
print("ExpSineSquared kernel parameters (detailed):")
for param, value in kernel_1.get_params().items():
    print(f"{param}: {value}")

print(f"\nWhiteKernel parameters:\n{kernel_2}\n")
print("WhiteKernel parameters (detailed):")
for param, value in kernel_2.get_params().items():
    print(f"{param}: {value}")

print("\nGaussianProcessRegressor parameters:")
for param, value in gp.get_params().items():
    print(f"{param}: {value}")

# 데이터 생성
np.random.seed(0)
X_train = np.random.rand(10, 1)
y_train = np.sin(X_train * 2 * np.pi).ravel()

# 사용자 정의 커널 및 GPR 구현
kernel = SumKernel(ExpSineSquaredKernel(length_scale=3.0, periodicity=3.0), WhiteKernel(noise_level=0.1))
gpr = GaussianProcessRegressorCustom(kernel)
gpr.fit(X_train, y_train)

# 예측
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_mean, y_std = gpr.predict(X_test)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'r.', markersize=10, label='Training Data')
plt.plot(X_test, y_mean, 'b-', label='Predicted Mean')
plt.fill_between(X_test.ravel(), y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='Confidence Interval (1 std dev)')
plt.title('Custom Gaussian Process Regression')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()

# 학습된 커널 매개변수 출력
print("Learned kernel parameters:")
print(f"ExpSineSquaredKernel length_scale: {kernel.k1.length_scale}")
print(f"ExpSineSquaredKernel periodicity: {kernel.k1.periodicity}")
print(f"WhiteKernel noise_level: {kernel.k2.noise_level}")
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