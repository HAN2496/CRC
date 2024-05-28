import os
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from scipy.interpolate import interp1d

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, predict_start, collect_end, num_points=500):
    if predict_start >= collect_end:
        return 0

    min_x = max(min(X_actual), min(X_pred))
    max_x = min(max(X_actual), max(X_pred))
    x_common = np.linspace(min_x, max_x, num_points)
    
    interpolator_actual = interp1d(X_actual, y_actual, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolator_pred = interp1d(X_pred, y_pred, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    y_actual_interp = interpolator_actual(x_common)
    y_pred_interp = interpolator_pred(x_common)

    mse = np.mean(np.square((y_actual_interp - y_pred_interp)))
    return mse


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

subject = Subject(6)
random_test_num = subject.get_random_test_num()
original_datas = subject.datas[random_test_num]
gp = GP(data_num=len(subject.datas[0]['header']))
interval = 20
corrected_datas =[]

# Initialize variables for data processing
t = 0
idx = 0
data_window = []
corrected_datas = []

# Assuming 'heelstrike' values are in original_datas and you can access the last value as needed
if len(original_datas['heelstrike']) > 0:
    X_scalar_end = original_datas['heelstrike'][-1]  # or however you calculate the end scalar

while idx < len(original_datas['header']):
    if t <= 4:
        data_point = {
            'header': original_datas['header'][idx],
            'hip_sagittal': original_datas['hip_sagittal'][idx],
            'heelstrike': original_datas['heelstrike'][idx],
            'heelstrike_x': original_datas['heelstrike_x'][idx],
            'heelstrike_y': original_datas['heelstrike_y'][idx],
            'torque': original_datas['torque'][idx]
        }
        corrected_datas.append(data_point)
        X_scalar_end = original_datas['heelstrike'][idx]  # Update X_scalar_end as the last known heelstrike value
    else:
        data_window.append(original_datas['hip_sagittal'][idx])
        if len(data_window) == interval:
            current_prediction = gp.find_pred_value_by_scalar(X_scalar_end)
            gp.translation(delx=0, dely=current_prediction - original_datas['hip_sagittal'][idx])

            X_actual = np.array([d['heelstrike'] for d in corrected_datas])
            y_actual = np.array([d['hip_sagittal'] for d in corrected_datas])
            X_pred, y_pred = gp.predict(np.column_stack((X_actual, y_actual)))

            predict_start = X_actual.min()
            collect_end = X_actual.max()

            contour_error = calculate_contour_error(X_actual, y_actual, X_pred, y_pred, predict_start, collect_end)

            scales = gradient_descent_algorithm(1.0, 1.0, 0.001, 10, (X_actual, y_actual))

            data_window.clear()

    t += 0.02
    idx += 1
    if t >= 10:
        break
