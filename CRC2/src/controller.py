import os
import torch
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from utils.td import TD
from utils.datasets import Datasets
from src.trajectory_generator import Reference

from configs.config_control import *

# import matplotlib
# matplotlib.use('Agg')

class Controller:
    def __init__(self, subject, reference):
        self.reference = reference
        self.subject = subject
        self.control_interval = 1
        self.scale_histories = []
        self.scale_x_before, self.scale_y_before = 1.0, 1.0
    
    def control(self, subject_datas, torques):
        scale_x, scale_y = self.minimize_scale(subject_datas['heelstrike'][-1], subject_datas['header'], subject_datas['hip_sagittal'])
        X_gp, y_reference = self.reference.scale(subject_datas['heelstrike'][-1], subject_datas['header'][-1], scale_x, scale_y, scale_from_end=False)
        error_before = y_reference[1] - subject_datas['hip_sagittal'][-1]
        v_t = subject_datas['hip_sagittal_v'][-1]
        x_t = subject_datas['hip_sagittal'][-1]
        print(f"scale: {scale_x, scale_y}")

        desired_kinematics = []

        for i in range(self.control_interval):
            error =  y_reference[i+1] - x_t
            d_error = (error - error_before) / dt
            total_error = Kp * error + Kd * d_error
            torque_input = total_error + torques[i]

            a_t = self.subject.move(torque_input)
            v_t1 = v_t + a_t * dt
            x_t1 = x_t + v_t1 * dt # + 0.5 * a_t * dt ** 2

            desired_kinematics.append([x_t1, v_t1, a_t])

            x_t = x_t1
            v_t = v_t1
            error_before = error
        return desired_kinematics, scale_x, scale_y

    def minimize_scale(self, heelstrike, headers, hip_sagittals, plot_scale_process = False):
        detail_scale_histories = []

        def objective(params, X, y, heelstrike):
            scale_x, scale_y = params
            X_gps, y_gps = self.reference.scale(heelstrike, X[-1], scale_x, scale_y)
            detail_scale_histories.append([scale_x, scale_y])
            error = 0
            for X_gp, y_gp in zip(X_gps, y_gps):
                if X_gp >= X[0]:
                    closest_idx = np.abs(X_gp - X).argmin()
                    error += (y_gp - y[closest_idx]) ** 2
            if plot_scale_process:
                pass
                # ax.cla()
                # ax.plot(X, y)
                # ax.plot(X_gps, y_gps, linestyle='--')
                # ax.text(0.5, 0.5, f"scale: {scale_x, scale_y}", transform=ax.transAxes, fontsize=12, verticalalignment='center')
                # plt.pause(0.5)
                pass

            return error
        initial_params = [self.scale_x_before, self.scale_y_before]

        X = headers
        y = hip_sagittals
        bounds = [(0.5, 2.0), (0.5, 2.0)]
        result = minimize(objective, initial_params, args=(X, y, heelstrike), method='Powell', bounds=bounds)
        if result.success:
            pass
        else:
            print("Optimization failed:", result.message)

        self.scale_histories.append([heelstrike, X[-1], result.x[0], result.x[1], detail_scale_histories])
        self.scale_x_before, self.scale_y_before = result.x
        return result.x




if __name__ == "__main__":
    control = Controller()
    control.control()
    #control.plot()
