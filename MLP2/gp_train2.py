from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump

def find_zero_indices(values_list, target_value):
    return [index for index, value in enumerate(values_list) if value == target_value]


interval = 60
hip_angles, heel_radii, heel_positions = load_data()


total_samples = len(hip_angles)
print(f"Total data length: {total_samples}")

initial_hip_angles, mid_hip_angles, final_hip_angles = [], [], []
initial_heel_positions, mid_heel_positions, final_heel_positions = [], [], []
initial_heel_radii, mid_heel_radii, final_heel_radii = [], [], []
zero_heel_indices = []

for i in range(total_samples):
    current_hip_angles = hip_angles[i]
    current_heel_positions = heel_positions[i]
    current_heel_radii = heel_radii[i]
    zero_indices = find_zero_indices(heel_positions[i], 0)
    zero_heel_indices.append(zero_indices)

    initial_hip_angles.append(current_hip_angles[zero_indices[0]:zero_indices[1]])
    initial_heel_positions.append(current_heel_positions[zero_indices[0]:zero_indices[1]])

    mid_hip_angles.append(current_hip_angles[zero_indices[2]:zero_indices[-1]])
    mid_heel_positions.append(current_heel_positions[zero_indices[2]:zero_indices[-1]])
    mid_heel_radii.append((current_heel_radii[zero_indices[2]:zero_indices[-1], :]))

    final_hip_angles.append(current_hip_angles[zero_indices[-1]:-1])
    final_heel_positions.append(current_heel_positions[zero_indices[-1]:-1])

# Using mid heel radii for regression
X_original = mid_heel_radii[0]
y_original = mid_hip_angles[0]
X = X_original[:-interval]
y = y_original[interval:]

plt.plot(range(len(y_original)), y_original)
plt.plot(range(len(y)), y)
plt.show()

kernel = ExpSineSquared(length_scale=3.0, periodicity=3.0,
                        length_scale_bounds=(0.1, 100.0),
                        periodicity_bounds=(0.1, 100.0)
                        ) + WhiteKernel(noise_level=0.1)

gp_model = GaussianProcessRegressor(kernel=kernel)
gp_model.fit(X, y)
#dump(gp_model, 'gaussian_process_regressor.joblib')

# Predicting using the same data for demonstration
predicted_hip_angles, uncertainty = gp_model.predict(X, return_std=True)

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_original)), y_original, 'r.', markersize=10, label='Actual Data (Hip Angles)')
plt.plot(range(len(y)), predicted_hip_angles, 'b-', label='Predicted Data (Hip Angles)')
plt.fill_between(range(len(y)), predicted_hip_angles - uncertainty, predicted_hip_angles + uncertainty, color='blue', alpha=0.2,
                 label='Confidence Interval (1 std dev)')
plt.title('Comparison of Actual and Predicted Hip Angles')
plt.xlabel('Time')
plt.ylabel('Hip Sagittal Angle')
plt.legend()
plt.show()
