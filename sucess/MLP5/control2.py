import os
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
import matplotlib.animation as animation
from IPython import display

def calculate_contour_error(X_actual, y_actual, X_pred, y_pred, predict_start, predict_end):
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
    translation_history = []
    translation_history.append((X_scalar_end, y_end))
    for i in range(num_iterations):
        X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y, X_scalar_end, y_end)
        plt.plot(X_actual, y_actual, 'r.', label='subject trajectory')
        if i % 20 == 0:
            tmp = len(X_actual)
            plt.plot(X_actual, y_pred_new[:tmp], 'k')
            plt.legend()
            plt.pause(0.001)
            plt.cla()
            print(y_pred_new[tmp], y_end)
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

    return scale_x, scale_y, scales_history, translation_history


subject_original = Subject(6, cut=True)
subject = Subject(6, cut=True)
original_datas = subject.datas

gp = GP(subject)
interval = 20
corrected_datas =[]

show = False
if show:
    plt.plot(original_datas['header'], original_datas['heelstrike'])
    #plt.plot(original_datas['header'], original_datas['hip_sagittal_speed'])
    #plt.plot(original_datas['header'], original_datas['hip_sagittal_acc'])
    plt.show()


idx = 0
t = original_datas['header'][0]
dt = original_datas['header'][1] - t
initial_t = t
heelstrike = original_datas['heelstrike'][0]
data_window = []
first_time = True
error = 0
Kp = 100
tmp = 0
idx_contrl_start=0
section = 0
sections = []
scale_histories = []
translation_histories = []

collect_num = 0.5 * 200

print(original_datas['heelstrike'])

while idx < len(original_datas['header'])-1:

    if idx <= collect_num:
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
        corrected_datas.append(data_point)
    else:
        if first_time:
            print("data collecting finished")
            sections.append(idx)
            idx_contrl_start = idx
            first_time = False
            idx = idx - 1
        #current_prediction = gp.find_pred_value_by_scalar(corrected_datas[-1]['heelstrike'], section)
        current_prediction = gp.find(idx)
        gp.translation(delx=0, dely=current_prediction - original_datas['hip_sagittal'][idx])
        if original_datas['hip_sagittal'][idx] != gp.y_pred[idx]:
            print('asdf')
        if original_datas['heelstrike'][idx] != corrected_datas[-1]['heelstrike']:
            print('something wrong..')
            print('heelstrike', original_datas['heelstrike'][idx], corrected_datas[-1]['heelstrike'])
        #print('sagittal', original_datas['hip_sagittal'][idx], corrected_datas[-1]['hip_sagittal'])
        data_gd = ([corrected_datas[i]['heelstrike'] for i in range(idx)],
                   [corrected_datas[i]['hip_sagittal'] for i in range(idx)])
        scale_x, scale_y, scale_history, translation_history = gradient_descent_algorithm(0.001, 200, data_gd, gp)
        scale_histories.append(scale_history)
        translation_histories.append(translation_history)
        #X_scalar_gp, y_pred_gp = gp.scale(scale_x, scale_y, corrected_datas[-1]['heelstrike'], corrected_datas[-1]['hip_sagittal'])
        X_scalar_gp, y_pred_gp = gp.scale(scale_x, scale_y, corrected_datas[-1]['heelstrike'], corrected_datas[-1]['hip_sagittal'])
        torque_gp = subject.calc_torque(y_pred_gp)
        torque_subject = original_datas['torque'][idx]

        #plt.plot(X_scalar_gp[:-1], y_pred_gp[:-1], 'b.', label='gp after scale')

        #display.clear_output(wait=True)
        error = y_pred_gp[idx] - corrected_datas[-1]['hip_sagittal']
        tmp = corrected_datas[-1]['hip_sagittal']
        tmp2 = original_datas['hip_sagittal'][idx]
        torque_input = Kp * error + torque_subject

        a_t = subject.move(torque_input)[0]
        a_dt = a_t - corrected_datas[-1]['hip_sagittal_acc']
        x_t = corrected_datas[-1]['hip_sagittal']
        v_t = corrected_datas[-1]['hip_sagittal_speed']
        x_t1 = x_t + v_t * dt
        v_t1 = v_t + a_t * dt
        a_t1 = a_t + a_dt
        if original_datas['heelstrike'][idx+1] == 0 and idx != 0:
            section += 1
            print(f'section will change to {section}...')
            v_t1 = original_datas['hip_sagittal_speed'][idx+1]
            a_t1 = original_datas['hip_sagittal_acc'][idx+1]
            torque_input = original_datas['torque'][idx+1]
            sections.append(idx+1)

        data_point = {
            'header': original_datas['header'][idx+1],
            'hip_sagittal': x_t1,
            'hip_sagittal_speed': v_t1,
            'hip_sagittal_acc': a_t1,
            'heelstrike': original_datas['heelstrike'][idx+1],
            'heelstrike_x': original_datas['heelstrike_x'][idx+1],
            'heelstrike_y': original_datas['heelstrike_y'][idx+1],
            'torque': torque_input
        }
        corrected_datas.append(data_point)

    t += 0.005
    idx += 1
    if t >= 30:
        break

scale_histories = np.array(scale_histories)
translation_histories = np.array(translation_histories).squeeze(1)

print("="*30)
print(f"Length of Total dataset: {len(original_datas['header'])}")
print(f"Scale history shape: {scale_histories.shape}")
print(f"Translation history shape: {translation_histories.shape}")
print(f"control start idx: {idx_contrl_start}")
print(f"sections: {sections}")
print("="*30)
print("Finish")


if show:
    plt.plot(original_datas['header'], original_datas['hip_sagittal'], label='original trajectory')
    plt.plot([corrected_datas[i]['header'] for i in range(idx)], [corrected_datas[i]['hip_sagittal'] for i in range(idx)], 'r', label='corrected trajectory')
    dely = gp.y_pred[idx_contrl_start] - original_datas['hip_sagittal'][idx_contrl_start]
    #gp.translation(dely=dely)
    plt.plot(original_datas['header'], gp.y_pred, 'b', label='gp data')
    plt.axvline(original_datas['header'][idx_contrl_start-1], linestyle='--')
    for section in sections:
        plt.axvline(original_datas['header'][section], linestyle='--', color='black')
    plt.legend()
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
subject_origin, = plt.plot([], [], 'ro', label='Original Subject Trajectory')
subject_corrected, = plt.plot([], [], 'go', label='Corrected Subject Trajectory')
gp_origin, = plt.plot([], [], 'b', label='Original GP Trajectory')
gp_corrected, = plt.plot([], [], 'm--', label='GP Prediction during Scaling')

def init():
    ax.set_xlim(min(original_datas['header']), max(original_datas['header']))
    ax.set_ylim(min(original_datas['hip_sagittal']) - 1, max(original_datas['hip_sagittal']) + 1)
    ax.legend()
    return subject_origin, subject_corrected, gp_origin, gp_corrected

def update(frame):
    if frame <= idx_contrl_start:
        frame2 = frame
        scale_history = scale_histories[frame2][0]
        translation_history = translation_histories[frame2]
    else:
        frame2 = int((frame - idx_contrl_start - 1) / 500) + idx_contrl_start
        frame3 = (frame - idx_contrl_start - 1) % 500
        scale_history = scale_histories[frame2][frame3]
        translation_history = translation_histories[frame2]
        print(f"frame: {frame}, frame2: {frame2}, frame3: {frame3}")

    #gp scaling 보여주는 곳
    scale_x, scale_y, _ = scale_history
    translation_x, translation_y = translation_history
    X_gp, y_gp = gp.scale(scale_x, scale_y, corrected_datas[frame2]['header'], corrected_datas[frame2]['hip_sagittal'])
    tmp = frame2 + 100
    gp_corrected.set_data(original_datas['header'][:tmp], y_gp[:tmp])

    #gp origin 라인 플롯
    gp_origin.set_data(original_datas['header'], gp.y_pred)

    #subject origin 플롯
    subject_origin.set_data(original_datas['header'][:frame2], original_datas['hip_sagittal'][:frame2])

    #control 되고 있는 subject 플롯
    xdata_corr = [d['header'] for d in corrected_datas[:frame2]]
    ydata_corr = [d['hip_sagittal'] for d in corrected_datas[:frame2]]
    subject_corrected.set_data(xdata_corr, ydata_corr)
    ax.set_xlim(min(original_datas['header']), original_datas['header'][tmp])

    return subject_origin, subject_corrected, gp_origin, gp_corrected

list1 = list(range(0, idx_contrl_start + 1))
list2 = list(range(idx_contrl_start + 1, len(original_datas['header'])*500, 10))
frames = list1 + list2
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
plt.show()










"""
line1: orginial data
line2: scaled gp data
line3: scale 이전, 즉, translation만 된 data
line4: 수정된 trajectory



"""
"""
fig, ax = plt.subplots()
ln_orig, = plt.plot([], [], 'r', label='Original Trajectory')
ln_corr, = plt.plot([], [], 'g', label='Corrected Trajectory')
ln_gp, = plt.plot([], [], 'b', label='GP Predicted Trajectory')

def init():
    ax.set_xlim(min(original_datas['header']), max(original_datas['header']))
    ax.set_ylim(min(original_datas['hip_sagittal']) - 1, max(original_datas['hip_sagittal']) + 1)
    ax.legend()
    return ln_orig, ln_corr, ln_gp

def update(frame):
    ln_orig.set_data(original_datas['header'][:frame], original_datas['hip_sagittal'][:frame])
    
    xdata_corr = [d['header'] for d in corrected_datas[:frame]]
    ydata_corr = [d['hip_sagittal'] for d in corrected_datas[:frame]]
    ln_corr.set_data(xdata_corr, ydata_corr)
    
    if frame < len(gp.y_pred):
        ln_gp.set_data(original_datas['header'][:frame], gp.y_pred[:frame])
    
    return ln_orig, ln_corr, ln_gp

ani = animation.FuncAnimation(fig, update, frames=len(original_datas['header']),
                    init_func=init, blit=True)

ani.save('test.gif', writer='imagemagick', fps=30, dpi=100)
plt.show()
"""








"""
def update_frame(i, scale_history, translation_histories, X_actual, y_actual, line1, line2, line3, title):
    scale_x, scale_y, _ = scale_history[i]
    translation_x, translation_y = translation_histories[i]
    X_scalar_pred_new, y_pred_new = gp.scale(scale_x, scale_y, translation_x, translation_y)
    X_scalar_pred_new2, y_pred_new2 = gp.scale(1.0, 1.0, translation_x, translation_y)
    line1.set_data(X_actual, y_actual)
    line2.set_data(X_scalar_pred_new, y_pred_new)
    line3.set_data(X_scalar_pred_new2, y_pred_new2)
    title.set_text(f'Iteration {i}: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}')
    return line1, line2


fig, ax = plt.subplots()
line1, = ax.plot([], [], 'k.', markersize=10, label='Actual Data')
line2, = ax.plot([], [], 'b-', label='Predicted Data')
line3, = ax.plot([], [], 'g-', label='Initial GP Data')
line4, = ax.plot([], [], 'r.', label='Corrected Data')
title = ax.set_title('')
ax.legend()
ax.set_xlim(min(original_datas['header']), 100)
ax.set_ylim(min(original_datas['hip_sagittal']) - 10, max(original_datas['hip_sagittal']) + 10)

#interval이 속도 결정하는듯. 작을수록 빠름
frames=range(0, len(scale_histories), 1) #<- 이렇게 하면 2개씩만
ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=2, fargs=(scale_histories, translation_histories, original_datas['header'], original_datas['hip_sagittal'], line1, line2, line3, line4, title))

writervideo = animation.FFMpegWriter(fps=60)

ani.save('gradient_descent_plot.gif', writer='imagemagick', fps=30, dpi=100)
plt.show()"""