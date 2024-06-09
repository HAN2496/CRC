import os
import time
import pandas as pd
import numpy as np
from subject import Subject
from gp import GP
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from IPython import display
from datasets import Datasets
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import imageio

class Control:
    def __init__(self):
        self.subject = Subject(6, cut=True)
        self.gp = GP(self.subject)
        self.original_datas = self.subject.datas
        self.dt = self.original_datas['header'][1] - self.original_datas['header'][0]
        self.total_data_num = len(self.original_datas['heelstrike'])
        print("Total data num: ", self.total_data_num)
        self.original_datas.appends({'start_indices': self.subject.start_indices})
        self.corrected_datas = Datasets()

        self.Kp = 1000
        self.learning_rate = 0.002
        self.num_iterations = 201

    def control(self):
        idx = 0
        interval_predict = 10
        filenames = []
        while idx < self.total_data_num - interval_predict:
            if self.original_datas['section'][idx] == 0:
                self.corrected_datas.appends(self.original_datas.indexs(idx))
                idx += 1
            else:
                print(f"Now idx: {idx} (total: {self.total_data_num})")
                plt.figure(figsize=(20, 15))
                plt.plot(self.original_datas['header'], self.gp.y_pred_original, color='blue', label='gp original')
                # plt.fill_between(self.original_datas['header'], self.gp.y_pred_original - self.gp.sigma_original, self.gp.y_pred_original + self.gp.sigma_original,
                #                  color='blue', alpha=0.2, label='Confidence Interval (1 std dev)')

                y_pred = self.gp.find(idx-1)
                plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'],  color='green', label='original subject')

                plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], marker='.', linestyle='', color='k', label='corrected subject')
                dely = y_pred - self.corrected_datas['hip_sagittal'][-1]
                self.gp.translation(delx=0, dely=dely)
                # plt.legend()
                # plt.show()
                # tic = time.time()
                # x_scale, y_scale, history_scale, history_translation = self.gradient_descent_algorithm(idx)
                # toc = time.time()
                # print('gd', toc - tic)
                # tic = time.time()
                x_scale, y_scale = self.minimize_scale(idx)
                # toc = time.time()
                # print('minimize', toc - tic)
                _, y_pred_gp = self.gp.scale(x_scale, y_scale, self.corrected_datas['heelstrike'][-1],
                                            self.corrected_datas['hip_sagittal'][-1])
                plt.plot(self.corrected_datas['header'], self.gp.y_pred[:idx], color='red', label='gp scaled')
                plt.legend()
                filename = f"tmp/{idx}.png"
                filenames.append(filename)
                plt.savefig(filename)
                plt.close()

                #plt.pause(0.001)
                #plt.cla()
                #plt.close()
                torque_gp = self.subject.calc_torque(y_pred_gp[:idx])[-1]
                torque_subject = self.original_datas['torque'][idx:idx+interval_predict]
                error = y_pred_gp[idx:idx+interval_predict] - y_pred_gp[idx+1:idx+interval_predict+1]
                torque_input = self.Kp * error + torque_subject

                a_t = self.subject.move(torque_input=torque_input)
                x_t = self.corrected_datas['hip_sagittal'][-1]
                v_t = self.corrected_datas['hip_sagittal_speed'][-1]
                a_t_prev = self.corrected_datas['hip_sagittal_acc'][-1]
                for i in range(interval_predict):                    
                    #x_t1 = x_t + v_t * self.dt
                    x_t1 = x_t + v_t * self.dt + 0.5 * a_t_prev * self.dt**2
                    v_t1 = v_t + (a_t[i] + a_t_prev) / 2 * self.dt
                    a_t1 = (a_t[i] + a_t_prev) / 2

                    data_point = {
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
                    a_t_prev = a_t[i]
                    idx += 1
    
                self.gp.translation(delx=0, dely=-dely)

        frames = []
        print('now gif will save ...')
        existing_gif_count = sum(1 for file in os.listdir("image/") if file.startswith("output") and file.endswith(".gif"))
        next_gif_number = existing_gif_count + 1
        exportname = f"image/output{next_gif_number}_K{self.Kp}.gif"
        duration_rate = 1
        for filename in filenames:
            if filename.endswith(".png"):
                frames.append(imageio.imread(filename))
        imageio.mimsave(exportname, frames, format='GIF', duration=duration_rate)
        plt.savefig(exportname, filenames[-1])

        for filename in set(filenames):
            os.remove(filename)
        print('gif saved.')

    def calculate_contour_error(self, y, y_pred, interval=0):
        mse = np.mean(np.square(y[interval:] - y_pred[interval:]))
        return mse

    def minimize_scale(self, idx):
        def objective(params, X, y, gp, idx, interval=0):
            scale_x, scale_y = params
            _, y_pred = gp.scale(scale_x, scale_y, X[-1], y[-1], idx=idx)
            error = np.mean(np.square(y[interval:] - y_pred[interval:]))
            return error
        initial_params = [1.0, 1.0]
        X = self.corrected_datas['heelstrike']
        y = self.corrected_datas['hip_sagittal']
        result = minimize(objective, initial_params, args=(X, y, self.gp, idx), method='L-BFGS-B')
        if result.success:
            optimized_scale_x, optimized_scale_y = result.x
            #print("Optimization successful:", result.message)
            print("Optimized scales:", optimized_scale_x, optimized_scale_y)
        else:
            raise ValueError("Optimization failed:", result.message)

        return result.x

    def gradient_descent_algorithm(self, idx):
        print_result = 0
        epsilon = 0.005
        scale_x, scale_y = 1.0, 1.0
        X = self.corrected_datas['heelstrike']
        y = self.corrected_datas['hip_sagittal']
        scales_history = []
        translation_history = []
        translation_history.append((X[-1], y[-1]))
        for i in range(self.num_iterations):
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x, scale_y, X[-1], y[-1], idx=idx)
            #print(f"scale: {scale_x, scale_y}, y: {y[-1]}, y_pred: {y_pred[-1]},")
            # plt.plot(self.corrected_datas['header'], y, 'r.', label='subject trajectory')
            # if i % 20 == 0:
            #     plt.plot(self.corrected_datas['header'], y_pred, 'k')
            #     plt.pause(0.001)
            #     plt.cla()
            #     pass
            contour_error = self.calculate_contour_error(y, y_pred)
            scales_history.append((scale_x, scale_y, contour_error))

            if print_result % 100 == 0:
               print(f"Iteration {i}: Contour Error = {contour_error}, Scale_X = {scale_x}, Scale_Y = {scale_y}")
        
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x + epsilon, scale_y, X[-1], y[-1], idx=idx)
            #print(f"scale1: {scale_x, scale_y}, y: {y[-1]}, y_pred: {y_pred[-1]},")
            error_x_plus = self.calculate_contour_error(y, y_pred)
            X_scalar_pred_new, y_pred = self.gp.scale(scale_x - epsilon, scale_y, X[-1], y[-1], idx=idx)
            #print(f"scale2: {scale_x, scale_y}, y: {y[-1]}, y_pred: {y_pred[-1]},")
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
control.control()
#control.plot()
#for idx in range(control.total_data_num - 1):
#for idx in range(150):
#    control.control(idx)
