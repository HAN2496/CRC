import os
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

"""
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
"""

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, predict_start, predict_end, num_points=500):
    if predict_start >= predict_end:
        return 0
    x_common = np.linspace(predict_start, predict_end, num_points)
    
    interpolator_actual = interp1d(X_actual, y_actual, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolator_pred = interp1d(X_pred, y_pred, kind='linear', bounds_error=False, fill_value="extrapolate")

    y_actual_interp = interpolator_actual(x_common)
    y_pred_interp = interpolator_pred(x_common)

    mse = np.mean(np.square((y_actual_interp - y_pred_interp)))
    return mse

def gradient_descent_algorithm(learning_rate, num_iterations, data, gp):
    print_result = 0
    epsilon = 0.005
    scale_x = 1.0
    scale_y = 1.0
    X_actual, y_actual = data
    X_scalar_end = X_actual[-1]
    y_end = y_actual[-1]
    predict_start = X_actual[0]
    predict_end = X_scalar_end

    scales_history = []
    for i in range(num_iterations):
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y, predict_end, y_end)
        contour_error = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new, predict_start=predict_start, predict_end=predict_end)
        scales_history.append((scale_x, scale_y, contour_error))

        #if print_result % 50 == 0:
        #    print(f"Iteration {i}: Contour Error = {contour_error}, Scale_X = {scale_x}, Scale_Y = {scale_y}")
    
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x + epsilon, scale_y, X_scalar_end, y_end)
        error_x_plus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new, predict_start=predict_start, predict_end=predict_end)
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x - epsilon, scale_y, X_scalar_end, y_end)
        error_x_minus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new, predict_start=predict_start, predict_end=predict_end)
        gradient_x = (error_x_plus - error_x_minus) / (2 * epsilon)

        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y + epsilon, X_scalar_end, y_end)
        error_y_plus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new, predict_start=predict_start, predict_end=predict_end)
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y - epsilon, X_scalar_end, y_end)
        error_y_minus = calculate_contour_error(X_actual, y_actual, X_scalar_pred_new, y_pred_new, predict_start=predict_start, predict_end=predict_end)
        gradient_y = (error_y_plus - error_y_minus) / (2 * epsilon)

        scale_x -= learning_rate * gradient_x
        scale_y -= learning_rate * gradient_y
        print_result +=1

    return scale_x, scale_y, scales_history




subject_original = Subject(6, cut=True)
subject = Subject(6, cut=True)
random_test_num = subject_original.get_random_test_num()
original_datas = subject_original.datas
gp = GP(original_datas)
interval = 20
corrected_datas ={}

idx = 0
t = original_datas['header'][0]
initial_t = t
heelstrike = original_datas['heelstrike'][0]
data_window = []

while idx < len(original_datas['header']):
    data_point = {
        'header': original_datas['header'][idx],
        'hip_sagittal': original_datas['hip_sagittal'][idx],
        'heelstrike': original_datas['heelstrike'][idx],
        'heelstrike_x': original_datas['heelstrike_x'][idx],
        'heelstrike_y': original_datas['heelstrike_y'][idx],
        'torque': original_datas['torque'][idx]
    }
    if t <= initial_t + 5:
        corrected_datas[idx] = data_point
    else:
        corrected_datas[idx] = data_point
        #plt.plot([corrected_datas[i]['header'] for i in range(idx)], [corrected_datas[i]['hip_sagittal'] for i in range(idx)])
        #plt.show()
        current_prediction = gp.find_pred_value_by_scalar(corrected_datas[idx]['heelstrike'])
        gp.translation(delx=0, dely=current_prediction - original_datas['hip_sagittal'][idx])
        data_gd = ([corrected_datas[i]['heelstrike'] for i in range(idx)],
                   [corrected_datas[i]['hip_sagittal'] for i in range(idx)])
        final_scale_x, final_scale_y, scales_history = gradient_descent_algorithm(0.001, 500, data_gd, gp)

        """
        update_data_point = {
            'header': original_datas['header'][idx],
            'hip_sagittal': original_datas['hip_sagittal'][idx],
            'heelstrike': original_datas['heelstrike'][idx],
            'heelstrike_x': original_datas['heelstrike_x'][idx],
            'heelstrike_y': original_datas['heelstrike_y'][idx],
            'torque': original_datas['torque'][idx]
        }
        corrected_datas[idx] = update_data_point
        """

    t += 0.02
    idx += 1
    if t >= 30:
        break
