from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump

def find_all_indexes(lst, value):
    return [index for index, current in enumerate(lst) if current == value]

smooth = False
hips, heel_rads, heels = load_data(smooth=smooth)


data_len = len(hips)
print(f"Total data length: {data_len}")

hip_starts, hip_mids, hip_ends = [], [], []
heel_starts, heel_mids, heel_ends = [], [], []
heel_start_rads, heel_mid_rads, heel_end_rads = [], [], []
heel_0_idxs = []

for i in range(data_len):
    current_hips = hips[i]
    current_heels = heels[i]
    current_heel_rads = heel_rads[i]
    heel_0_idx = find_all_indexes(heels[i], 0)
    heel_0_idxs.append(heel_0_idx)

    hip_starts.append(current_hips[heel_0_idx[0]:heel_0_idx[1]])
    heel_starts.append(current_heels[heel_0_idx[0]:heel_0_idx[1]])

    hip_mids.append(current_hips[heel_0_idx[2]:heel_0_idx[-1]])
    heel_mids.append(current_heels[heel_0_idx[2]:heel_0_idx[-1]])
    heel_mid_rads.append((current_heel_rads[heel_0_idx[2]:heel_0_idx[-1], :]))

    hip_ends.append(current_hips[heel_0_idx[-1]:-1])
    heel_ends.append(current_heels[heel_0_idx[-1]:-1])

#X = heel_mids[0].reshape(-1, 1)
X = heel_mid_rads[0]
y = hip_mids[0]

kernel = ExpSineSquared(length_scale=3.0, periodicity=3.0
                        , length_scale_bounds=(0.1, 100.0),
                        periodicity_bounds=(0.1, 100.0)
                        ) + WhiteKernel(noise_level=0.1)


gp = GaussianProcessRegressor(kernel=kernel)

gp.fit(X, y)
dump(gp, 'gaussian_process_regressor.joblib')

# Predicting using the same data for demonstration
y_pred, sigma = gp.predict(X, return_std=True)

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