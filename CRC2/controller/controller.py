import os
import torch
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from datasets.subject import Subject
from datasets.td import TD
from controller.gp import GP
from datasets.datasets import Datasets

from configs.config_control import *

# import matplotlib
# matplotlib.use('Agg')

class Controller:
    def __init__(self, subject):
        self.gp = GP()
        self.subject = subject
        self.control_interval = 1
        self.scale_histories = []
    
    def control(self, subject_datas, heelstrikes, headers, hip_sagittals, torques):
        scale_x, scale_y = self.minimize_scale(subject_datas['heelstrike'], subject_datas['header'], subject_datas['hip_sagittal'])
        X_gp, y_gp = self.gp.scale(subject_datas['heelstrike'][-1], subject_datas['header'][-1], scale_x, scale_y)

        error_before = y_gp[1] - subject_datas['hip_sagittal'][-1]
        v_t = subject_datas['hip_sagittal_v'][-1]
        x_t = subject_datas['hip_sagittal'][-1]

        desired_kinematics = []
    
        for i in range(self.control_interval):
            error =  y_gp[i+1] - subject_datas['hip_sagittal'][-1]
            d_error = (error - error_before) / dt
            total_error = Kp * error + Kd * d_error
            torque_input = total_error + torques[i]

            a_t = self.subject.move(torque_input)
            v_t1 = v_t + a_t * dt
            x_t1 = x_t + v_t1 + dt + 0.5 * a_t * dt ** 2

            desired_kinematics.append([x_t1, v_t1, a_t])

            x_t = x_t1
            v_t = v_t1
            error_before = error
        return desired_kinematics

    def minimize_scale(self, heelstrikes, headers, hip_sagittals):
        detail_scale_histories = []
        def objective(params, X, y, heelstrike):
            scale_x, scale_y = params
            X_gps, y_gps = self.gp.scale(heelstrike, X[-1], scale_x, scale_y)
            detail_scale_histories.append([scale_x, scale_y])
            error = 0
            for X_gp, y_gp in zip(X_gps, y_gps):
                if X_gp >= X[0]:
                    closest_idx = np.abs(X_gp - X).argmin()
                    error += (y_gp - y[closest_idx]) ** 2
            return error
        initial_params = [1.0, 1.0]

        X = headers
        y = hip_sagittals
        heelstrike = heelstrikes[-1]
        result = minimize(objective, initial_params, args=(X, y, heelstrike), method='L-BFGS-B')#, bounds=bounds)
        if result.success:
            pass
        else:
            print("Optimization failed:", result.message)

        self.scale_histories.append([heelstrike, X[-1], result.x[0], result.x[1], detail_scale_histories])
        return result.x



class Control:
    def __init__(self):
        self.subject = TD(number=6)
        self.gp = GP()
        self.original_datas = self.subject.datas
        self.corrected_datas = Datasets()
        self.scale_histories = []
        self.total_data_num = len(self.original_datas['header'])
        print(f"total data num: {self.total_data_num}")

        self.control_interval = 1
        self.control_start_idx = 0

    def control(self):
        idx = 0
        collect_one_more = True
        while idx < self.total_data_num:
            if self.original_datas['section'][idx] == 0:
                """
                collect data at first loop
                """
                self.corrected_datas.appends(self.original_datas.indexs(idx))
                idx += 1
            else:
                if collect_one_more:
                    self.corrected_datas.appends(self.original_datas.indexs(idx))
                    self.gp.update(self.original_datas['header'][:idx], self.original_datas['heelstrike'])
                    collect_one_more = False
                    self.control_start_idx = idx
                    idx+=1
                print(f"Now idx: {idx} (total: {self.total_data_num})")

                scale_x, scale_y = self.minimize_scale()
                X_gp, y_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], self.corrected_datas['header'][-1],
                                           scale_x, scale_y)

                torque_subject = self.original_datas['torque'][idx:idx+self.control_interval]
                v_t = self.corrected_datas['hip_sagittal_v'][-1]
                x_t = self.corrected_datas['hip_sagittal'][-1]

                error_before = y_gp[1] - self.corrected_datas['hip_sagittal'][-1]
                for i in range(self.control_interval):
                    error =  y_gp[i+1] - self.corrected_datas['hip_sagittal'][-1]
                    d_error = (error - error_before) / dt
                    total_error = Kp * error + Kd * d_error
                    torque_input = total_error + torque_subject[i]

                    a_t = self.subject.move(torque_input)
                    v_t1 = v_t + a_t * dt
                    x_t1 = x_t + v_t1 + dt + 0.5 * a_t * dt ** 2

                    data_point = {
                        'section': self.original_datas['section'][idx],
                        'header': self.original_datas['header'][idx],
                        'hip_sagittal': x_t1,
                        'hip_sagittal_speed': v_t1,
                        'hip_sagittal_acc': a_t,
                        'heelstrike': self.original_datas['heelstrike'][idx],
                        'heelstrike_x': self.original_datas['heelstrike_x'][idx],
                        'heelstrike_y': self.original_datas['heelstrike_y'][idx],
                        'torque': torque_input
                    }
                    self.corrected_datas.appends(data_point)
                    x_t = x_t1
                    v_t = v_t1
                    error_before = error

                    idx += 1

    def minimize_scale(self):
        def objective(params, X, y, heelstrike):
            scale_x, scale_y = params
            X_gps, y_gps = self.gp.scale(heelstrike, X[-1], scale_x, scale_y)
            error = 0
            for X_gp, y_gp in zip(X_gps, y_gps):
                if X_gp >= X[0]:
                    closest_idx = np.abs(X_gp - X).argmin()
                    error += (y_gp - y[closest_idx]) ** 2
            return error
        initial_params = [1.0, 1.0]

        X = self.corrected_datas['header']
        y = self.corrected_datas['hip_sagittal']
        heelstrike = self.corrected_datas['heelstrike'][-1]
        result = minimize(objective, initial_params, args=(X, y, heelstrike), method='L-BFGS-B')#, bounds=bounds)
        if result.success:
            pass
        else:
            print("Optimization failed:", result.message)

        self.scale_histories.append([heelstrike, X[-1], result.x[0], result.x[1]])
        return result.x
    
    def plot_each(self, idx, save=True):
        plt.figure(figsize=(15, 10))
        plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'],  color='green', label='original subject')
        plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], marker='.', linestyle='', color='k', label='corrected subject')

        heelstrike, time, scale_x, scale_y = self.scale_histories[idx]
        X_pred_gp, y_pred_gp = self.gp.scale(heelstrike, time, scale_x, scale_y)
        plt.plot(X_pred_gp, y_pred_gp, color='blue', label='gp line for scaling')

        X_pred_gp, y_pred_gp = self.gp.scale(heelstrike, time, scale_x, scale_y, reverse=True)
        half_len = int(len(y_pred_gp))
        plt.plot(X_pred_gp[:half_len], y_pred_gp[:half_len], color='red', label='gp line for reference')
        plt.legend()
        if save:
            filename = f"tmp/{idx}.png"
            plt.savefig(filename)
        plt.close()
        return filename
    
    def plot(self):
        filenames = []
        for i in range(self.control_start_idx, self.total_data_num):
            filenames.append(self.plot_each(i))

        print('now png will change into gif ...')
    
        frames = []
        existing_gif_count = sum(1 for file in os.listdir("gifs/") if file.startswith("output") and file.endswith(".gif"))
        next_gif_number = existing_gif_count + 1
        exportname = f"gifs/output{next_gif_number}_Kp{Kp}_Kd_{Kd}"
        duration_rate = 1
        for filename in filenames:
            if filename.endswith(".png"):
                frames.append(imageio.imread(filename))
        imageio.mimsave(f"{exportname}.gif", frames, format='GIF', duration=duration_rate)
        plt.savefig(filenames[-1])

        # for filename in set(filenames):
        #     os.remove(filename)
        print('gif saved.')

if __name__ == "__main__":
    control = Control()
    control.control()
    #control.plot()
