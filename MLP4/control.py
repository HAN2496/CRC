import os
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, predict_start, predict_end, num_points=500):
    if predict_start >= predict_end:
        return 0
    mse = np.mean(np.square((y_actual - y_pred[:len(y_actual)])))
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
dt = original_datas['header'][1] - t
initial_t = t
heelstrike = original_datas['heelstrike'][0]
data_window = []
first_time = True
error = 0
Kp = 0
tmp = 0

def calc_angle(t1, t2, t3, x1, x2, x3):
    v1 = (x2 - x1) / (t2 - t1)
    v2 = (x3 - x2) / (t3 - t2)
    return (v2 - v1) / (t3 - t1)
while idx < len(original_datas['header']):
    data_point = {
        'header': original_datas['header'][idx],
        'hip_sagittal': original_datas['hip_sagittal'][idx],
        'hip_sagittal_speed': original_datas['hip_sagittal_speed'][idx],
        'hip_sagittal_acc': original_datas['hip_sagittal_acc'][idx],
        'heelstrike': original_datas['heelstrike'][idx],
        'heelstrike_x': original_datas['heelstrike_x'][idx],
        'heelstrike_y': original_datas['heelstrike_y'][idx],
        'torque': original_datas['torque'][idx]
    }
    if t <= initial_t + 2:
        corrected_datas[idx] = data_point
    else:
        corrected_datas[idx] = data_point
        current_prediction = gp.find_pred_value_by_scalar(corrected_datas[idx]['heelstrike'])
        gp.translation(delx=0, dely=current_prediction - original_datas['hip_sagittal'][idx])
        data_gd = ([corrected_datas[i]['heelstrike'] for i in range(idx)],
                   [corrected_datas[i]['hip_sagittal'] for i in range(idx)])
        scale_x, scale_y, scales_history = gradient_descent_algorithm(0.001, 500, data_gd, gp)
        #plt.plot(gp.X_scalar, gp.y_pred, 'g', label='gp before')
        gp.scale(scale_x, scale_y, corrected_datas[idx]['heelstrike'], corrected_datas[idx]['hip_sagittal'], save_data=True)
        print(f"idx: {idx}, scale_x: {scale_x}, scale_y: {scale_y}")
        #plt.plot([corrected_datas[i]['heelstrike'] for i in range(idx)], [corrected_datas[i]['hip_sagittal'] for i in range(idx)], 'r.', label='subject trajectory')
        #plt.plot(gp.X_scalar, gp.y_pred, 'b', label='gp after scale')
        #plt.axvline(original_datas['heelstrike'][idx])
        #plt.legend()
        #plt.show()
        torque_gp = subject.calc_torque(gp.y_pred)
        torque_subject = corrected_datas[idx]['torque']

        #plt.plot(gp.X_scalar, torque_gp, label='gp torque')
        #plt.plot([corrected_datas[i]['heelstrike'] for i in range(idx)], [corrected_datas[i]['torque'] for i in range(idx)], 'r', label='subject torque')
        #plt.legend()
        #plt.show()
        error = torque_gp[-1] - torque_subject
        #print('here', torque_gp, torque_subject, error)
        torque_input = Kp * error + torque_subject
        a_t = subject.move(torque_input)[0]
        x_t = corrected_datas[idx-1]['hip_sagittal'] + corrected_datas[idx-1]['hip_sagittal_speed'] * dt
        v_t = corrected_datas[idx-1]['hip_sagittal_speed'] + corrected_datas[idx-1]['hip_sagittal_acc'] * dt
        hip_sagittal_next = x_t + v_t * dt + 1/2 * a_t * dt * dt
        #x_t1 = corrected_datas[idx]['hip_sagittal'] + corrected_datas[idx]['hip_sagittal_speed'] * dt
        #v_t1 = corrected_datas[idx]['hip_sagittal_speed'] + a_t * dt
        #x_t2 = x_t1 + v_t1 * dt
        
        update_data_point = {
            'header': original_datas['header'][idx],
            'hip_sagittal': hip_sagittal_next,
            'hip_sagittal_speed': v_t,
            'hip_sagittal_acc': a_t,
            'heelstrike': original_datas['heelstrike'][idx],
            'heelstrike_x': original_datas['heelstrike_x'][idx],
            'heelstrike_y': original_datas['heelstrike_y'][idx],
            'torque': torque_input
        }
        if tmp % 2 == 0:
            corrected_datas[idx+1] = update_data_point
            idx += 1
        else:
            corrected_datas[idx+1] = update_data_point
            idx += 1

    t += 0.02
    idx += 1
    if t >= 30:
        break

plt.plot(original_datas['heelstrike'], original_datas['hip_sagittal'], label='original trajectory')
plt.plot([corrected_datas[i]['heelstrike'] for i in range(idx)], [corrected_datas[i]['hip_sagittal'] for i in range(idx)], 'r', label='corrected trajectory')
plt.plot(gp.X_scalar, gp.y_pred, 'b', label='gp data')
plt.legend()
plt.show()