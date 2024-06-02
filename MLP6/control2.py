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
from datasets import Datasets


class Control:
    def __init__(self):
        self.subject = Subject(6, cut=True)
        self.gp = GP(self.subject)
        self.original_datas = Datasets(self.subject.datas)
        self.total_data_num = len(self.original_datas['heelstrike'])
        print("Total data num: ", self.total_data_num)
        self.original_datas.appends({'start_indices': self.subject.start_indices})
        self.corrected_datas = Datasets()

        self.Kp = 1000
        self.learning_rate = 0.001
        self.num_iterations = 200
        self.collect_num = 0.5 * 200

    def control(self, idx):
        if idx < 0:
            raise ValueError
        if idx < self.collect_num:
            self.corrected_datas.appends(self.original_datas.indexs(idx))
        else:
            if idx == self.collect_num:
                print("Data collecting is finished.")


            y_pred = self.gp.find(idx)
            for i in [idx-2, idx-1]:
                print("original:", self.original_datas.index('hip_sagittal', i))
                print("corrected:", self.corrected_datas.index('hip_sagittal', i))

            # plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], label='correct')
            # plt.plot(self.corrected_datas['header'], self.gp.y_pred[:idx], label='gp')

            self.gp.translation(delx=0, dely=y_pred - self.corrected_datas.index('hip_sagittal', -1))
            #print("predicted:", self.gp.y_pred[idx])

            # plt.plot(self.corrected_datas['header'], self.gp.y_pred[:idx], label='gp moved')
            # plt.legend()
            # plt.show()
            self.gradient_descent_algorithm(idx)

            self.corrected_datas.appends(self.original_datas.indexs(idx))


    def calculate_contour_error(self, y, y_pred):
        mse = np.mean(np.square(y - y_pred))
        return mse

    def gradient_descent_algorithm(self, idx):
        print_result = 0
        epsilon = 0.005
        scale_x, scale_y = 1.0, 1.0
        X = self.corrected_datas.indexs(-1)['heelstrike']
        y = self.corrected_datas.indexs(-1)['hip_sagittal']

        scales_history = []
        translation_history = []
        translation_history.append((X[-1], y[-1]))
        for i in range(self.num_iterations):
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x, scale_y, X[-1], y[-1], idx=idx)
            print(f"1: {y_pred[-1]}")
            plt.plot(X, y, 'r.', label='subject trajectory')
            if i % 1 == 0:
                plt.plot(X, y_pred, 'k')
                plt.pause(0.001)
                plt.cla()
                #print(y_pred[tmp], y[-1])
            contour_error = self.calculate_contour_error(y, y_pred)
            scales_history.append((scale_x, scale_y, contour_error))

            #if print_result % 50 == 0:
            #    print(f"Iteration {i}: Contour Error = {contour_error}, Scale_X = {scale_x}, Scale_Y = {scale_y}")
        
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x + epsilon, scale_y, X[-1], y[-1], idx=idx)
            error_x_plus = self.calculate_contour_error(y, y_pred)
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x - epsilon, scale_y, X[-1], y[-1], idx=idx)
            error_x_minus = self.calculate_contour_error(y, y_pred)
            gradient_x = (error_x_plus - error_x_minus) / (2 * epsilon)

            X_scalar_pred_new, y_pred = self.gp.scale(scale_x, scale_y + epsilon, X[-1], y[-1], idx=idx)
            error_y_plus = self.calculate_contour_error(y, y_pred)
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x, scale_y - epsilon, X[-1], y[-1], idx=idx)
            error_y_minus = self.calculate_contour_error(y, y_pred)
            gradient_y = (error_y_plus - error_y_minus) / (2 * epsilon)

            scale_x -= self.learning_rate * gradient_x
            scale_y -= self.learning_rate * gradient_y
            print_result +=1

        return scale_x, scale_y, scales_history, translation_history

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
#control.plot()
for idx in range(control.total_data_num - 1):
    control.control(idx)
