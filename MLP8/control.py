import os
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from matplotlib import pyplot as plt
from datasets import Datasets
from scipy.optimize import minimize
import imageio
from matplotlib import animation

import matplotlib
matplotlib.use('Agg')

class Control:
    def __init__(self):
        self.subject = Subject(6, cut=True)
        self.gp = GP(self.subject)
        self.original_datas = self.subject.datas
        self.dt = self.original_datas['header'][1] - self.original_datas['header'][0]
        self.total_data_num = len(self.original_datas['header'])
        print("Total data num: ", self.total_data_num)
        self.original_datas.appends({'start_indices': self.subject.start_indices})
        self.corrected_datas = Datasets()

        self.Kp = 2000
        self.Ki = 0
        self.Kd = 100
        self.learning_rate = 0.001
        self.num_iterations = 201

    def control(self):
        one_more = True
        idx = 0
        interval_predict = 1
        filenames = []
        i_error = 0
        d_error = 0
        while idx < self.total_data_num - interval_predict:
            if self.original_datas['section'][idx] == 0:
                self.corrected_datas.appends(self.original_datas.indexs(idx))
                idx += 1
            else:
                if one_more:
                    self.corrected_datas.appends(self.original_datas.indexs(idx))
                    idx += 1
                    one_more = False
                    self.gp._init(self.original_datas['header'][:idx], self.original_datas['heelstrike'][:idx],
                                  self.original_datas['heelstrike_x'][:idx], self.original_datas['heelstrike_y'][:idx])
                    continue
                print(f"Now idx: {idx} (total: {self.total_data_num})")

                x_scale, y_scale = self.minimize_scale(idx)
                print(f"x scale: {x_scale} / y scale: {y_scale}")

        
                X_pred_gp, y_pred_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale,
                                                     self.corrected_datas['header'][-1], end=False)
                tmp_x, tmp_y = X_pred_gp, y_pred_gp

                torque_subject = self.original_datas['torque'][idx:idx+interval_predict]
                a_t_prev = self.corrected_datas['hip_sagittal_acc'][-1]
                v_t = self.corrected_datas['hip_sagittal_speed'][-1]
                x_t = self.corrected_datas['hip_sagittal'][-1]

                i_error = 0
                error_before = 0
                for i in range(interval_predict):
                    error =  y_pred_gp[i+1] - self.corrected_datas['hip_sagittal'][-1]
                    d_error = (error-error_before) / self.dt
                    i_error += error * self.dt
                    total_error =  self.Kp * error + self.Kd * d_error + self.Ki * i_error
                    torque_input = total_error + torque_subject[i]

                    #print('idx:', idx)
                    #print(f"error: {error}, subject torque: {torque_subject[i]}, torque_input: {torque_input}" )
                    a_subject = self.original_datas['hip_sagittal_acc'][idx]

                    #a_t = self.subject.move(torque_input=torque_input)[0]
                    a_t = a_subject + total_error
                    a_t1 = a_t
                    v_t1 = v_t + a_t * self.dt
                    x_t1 = x_t + v_t1 * self.dt + 0.5 * a_t * self.dt ** 2

                    #print(f'check move func: original " {round(a_subject, 3)} / move: {self.subject.move(torque_subject[i])[0]}')


                    #print(round(a_t, 3), round(v_t1, 3), round(x_t1, 3))
                    #print(round(v_t, 3), round(x_t, 3))

                    # a_t1 = (a_t + a_t_prev) / 2
                    # v_t1 = v_t + a_t1 * self.dt
                    # x_t1 = x_t + v_t * self.dt + 0.5 * a_t_prev * self.dt ** 2


                    plt.figure(figsize=(15, 10))
                    plt.ylim([-30, 30])
                    plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'],  color='green', label='original subject')
                    plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], marker='.', linestyle='', color='k', label='corrected subject')
                    X_pred_gp, y_pred_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale, self.corrected_datas['header'][-1])
                    plt.plot(X_pred_gp, y_pred_gp, color='blue', label='gp line for scaling')
                    # plt.text(self.corrected_datas['header'][-1]+0.5, self.corrected_datas['hip_sagittal'][-1], error)
                    # plt.text(self.corrected_datas['header'][-1]+0.5, self.corrected_datas['hip_sagittal'][-1] - 1, total_error)
                    # plt.text(self.corrected_datas['header'][-1]+0.5, self.corrected_datas['hip_sagittal'][-1] - 2, torque_subject[i])
                    half_len = int(len(tmp_y))
                    if i <= 1:
                        plt.plot(tmp_x[:half_len], tmp_y[:half_len], color='red', label='gp line for reference')
                    else:
                        plt.plot(tmp_x[:half_len], tmp_y[:half_len], color='red', marker='.', linestyle='', label='gp line for reference')
                    plt.legend()
                    filename = f"tmp/{idx}.png"
                    filenames.append(filename)
                    plt.savefig(filename)
                    plt.close()

                    data_point = {
                        'section': self.original_datas['section'][idx],
                        'header': self.original_datas['header'][idx],
                        'hip_sagittal': x_t1,
                        'hip_sagittal_speed': v_t1,
                        'hip_sagittal_acc': a_t1,
                        'heelstrike': self.original_datas['heelstrike'][idx],
                        'heelstrike_x': self.original_datas['heelstrike_x'][idx],
                        'heelstrike_y': self.original_datas['heelstrike_y'][idx],
                        'torque': torque_input
                    }
                    self.corrected_datas.appends(data_point)
                    x_t = x_t1
                    v_t = v_t1
                    a_t_prev = a_t
                    error_before = error

                    idx += 1

        frames = []
        print('now gif will save ...')
        existing_gif_count = sum(1 for file in os.listdir("image/") if file.startswith("output") and file.endswith(".gif"))
        next_gif_number = existing_gif_count + 1
        exportname = f"image/output{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}"
        duration_rate = 1
        for filename in filenames:
            if filename.endswith(".png"):
                frames.append(imageio.imread(filename))
        imageio.mimsave(f"{exportname}.gif", frames, format='GIF', duration=duration_rate)
        print(exportname, filenames[-1])
        plt.savefig(f"{exportname}.png", filenames[-1])

        # for filename in set(filenames):
        #     os.remove(filename)
        print('gif saved.')

    def minimize_scale(self, idx):
        def objective(params, X, y_actual, heelstrike, interval=0):
            tmp = 0
            scale_x, scale_y = params
            #print(f"x scale: {scale_x} / y scale: {scale_y}")
            X_pred, y_pred = self.gp.scale(heelstrike, scale_x, scale_y, X[-1])
            error = 0
            for x, y in zip(X_pred, y_pred):
                if x < X[0]:
                    continue
                distances = np.abs(x - X)
                closest_idxs = np.argsort(distances)[0]
                error += np.mean(np.square(y - self.corrected_datas['hip_sagittal'][closest_idxs]))
            if tmp % 20 == 0:
                # plt.plot(X_pred, y_pred, color='red')
                # plt.plot(self.corrected_datas['header'], y_actual, color='k')
                # plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'], color='b', linestyle='--')
                # plt.pause(0.001)
                # plt.cla()
                pass
            tmp+=1
            return error
        initial_params = [1.0, 1.0]

        X = self.corrected_datas['header']
        y = self.corrected_datas['hip_sagittal']
        #bounds = [(0.1, 10.0), (0.5, 1.5)]
        result = minimize(objective, initial_params, args=(X, y, self.corrected_datas['heelstrike'][-1]), method='L-BFGS-B')#, bounds=bounds)
        if result.success:
            optimized_scale_x, optimized_scale_y = result.x
            #print("Optimization successful:", result.message)
            # print("Optimized scales:", optimized_scale_x, optimized_scale_y)
        else:
            print("Optimization failed:", result.message)
            optimized_scale_x, optimized_scale_y = result.x
            #raise ValueError("Optimization failed:", result.message)

        return result.x

    def plot(self, show=True):
        x = self.original_datas['header']
        plt.plot(x, self.original_datas['hip_sagittal'], label='original trajectory')
        plt.plot(x, self.gp.y_pred_original, label='gp original')
        for idx in self.original_datas['start_indices']:
            plt.axvline(self.original_datas['header'][idx], color='black', linestyle='--')

        if show:
            plt.legend()
            plt.show()

control = Control()
control.control()
#control.plot()
#for idx in range(control.total_data_num - 1):
#for idx in range(150):
#    control.control(idx)
