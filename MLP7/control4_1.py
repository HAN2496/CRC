import os
import numpy as np
#from subject import Subject
from subject2 import Subject
from gp import GP
from matplotlib import pyplot as plt
from datasets import Datasets
from scipy.optimize import minimize
import matplotlib.animation as animation
import imageio
from scipy.signal import savgol_filter
# import matplotlib
# matplotlib.use('Agg')

class Control:
    def __init__(self):
        #self.subject = Subject(6, cut=True)
        self.subject = Subject()
        self.gp = GP(self.subject)
        self.gp_original = GP(self.subject)
        self.original_datas = self.subject.datas
        self.dt = self.original_datas['header'][1] - self.original_datas['header'][0]
        self.total_data_num = len(self.original_datas['header'])
        print("Total data num: ", self.total_data_num)
        self.original_datas.appends({'start_indices': self.subject.start_indices})
        self.corrected_datas = Datasets()
        self.scale_histories = []

        self.Kp = 2000
        self.Ki = 0
        self.Kd = 40
        self.learning_rate = 0.001
        self.num_iterations = 201

    def control(self):
        one_more = True
        save_data = False
        if save_data:
            unchange_datas = {
                "original subject": [],
                "corrected subject": [],
                "original gp": []
            }
            change_datas = {
                "corrected subject": [],
                "scaled gp": []
            }
        pass_high = 0
        idx = 0
        start_idx = 0
        interval_predict = 1
        total_idx = self.total_data_num - interval_predict
        filenames = []
        i_error = 0
        d_error = 0
        while idx < self.total_data_num - interval_predict:
            if self.original_datas['section'][idx] == 0:
                self.corrected_datas.appends(self.original_datas.indexs(idx))
                idx += 1
            else:
                if one_more:
                    start_idx = idx
                    self.start_idx = start_idx
                    self.corrected_datas.appends(self.original_datas.indexs(idx))
                    idx += 1
                    one_more = False
                    self.gp_original._init(self.original_datas['header'], self.original_datas['heelstrike'],
                                  self.original_datas['heelstrike_x'], self.original_datas['heelstrike_y'])
                    self.gp._init(self.original_datas['header'][:idx], self.original_datas['heelstrike'][:idx],
                                  self.original_datas['heelstrike_x'][:idx], self.original_datas['heelstrike_y'][:idx])
                    if save_data:
                        unchange_datas['original_subject'] = np.array([self.original_datas.datas])
                        unchange_datas['original_gp'] = np.array([
                            [self.gp_original.X_time_original],
                            [self.gp_original.y_pred_original]
                        ])
                    continue
                if pass_high < 62: #500:
                    start_idx = idx
                    self.corrected_datas.appends(self.original_datas.indexs(idx))
                    idx += 1
                    pass_high += 1
                    continue
                print(f"Now idx: {idx} (total: {self.total_data_num})")

                (x_scale, y_scale), detail_scale_hisotries = self.minimize_scale(idx)
                print(f"x scale: {x_scale} / y scale: {y_scale}")

                # tmp_x1, tmp_y1 = self.gp.scale(self.corrected_datas['heelstrike'][-1], 1.0, 1.0,
                #                                      self.corrected_datas['header'][-1], end=True)
                
                # plt.plot(tmp_x1, tmp_y1, label='before scaling')
                # tmp_x1, tmp_y1 = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale,
                #                                      self.corrected_datas['header'][-1], end=True)
            
                # plt.plot(tmp_x1, tmp_y1, label='after scaling')
                X_pred_gp, y_pred_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale,
                                                     self.corrected_datas['header'][-1], end=False)
                
                # plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], label='patient')
                # plt.legend()
                # plt.show()
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

                    # data = np.concatenate(np.array(self.corrected_datas['hip_sagittal']), np.array([x_t1]))
                    # smoothed_x_t1 = savgol_filter(data, window_length=5, polyorder=2)[-1]


                    #print(round(a_t, 3), round(v_t1, 3), round(x_t1, 3))
                    #print(round(v_t, 3), round(x_t, 3))

                    # a_t1 = (a_t + a_t_prev) / 2
                    # v_t1 = v_t + a_t1 * self.dt
                    # x_t1 = x_t + v_t * self.dt + 0.5 * a_t_prev * self.dt ** 2


                    plt.figure(figsize=(15, 10))
                    #plt.plot(self.gp_original.X_time_original, self.gp_original.y_pred_original, label='original gp line')
                    plt.plot(self.original_datas['header'][:idx], self.original_datas['hip_sagittal'][:idx],  color='black', label='original trajectory')
                    plt.plot(self.corrected_datas['header'][start_idx:], self.corrected_datas['hip_sagittal'][start_idx:], color='green', linewidth=3, label='corrected trajectory')
                    X_pred_gp, y_pred_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale, self.corrected_datas['header'][-1])
                    plt.plot(X_pred_gp, y_pred_gp, color='red', label='reference trajectory')
                    
                    half_len = int(len(tmp_y) / 4)
                    plt.plot(tmp_x[:half_len], tmp_y[:half_len], color='red', linestyle='--', label='control trajectory')
                    plt.legend(loc='lower left')
                    plt.xlim([min(X_pred_gp), max(tmp_x[:half_len])])
                    filename = f"tmp/{idx}.png"
                    filenames.append(filename)
                    plt.savefig(filename)
                    plt.close()

                    if idx == total_idx - 1:
                        existing_gif_count = sum(1 for file in os.listdir("gifs/") if file.startswith("Patient") and file.endswith(".gif"))
                        next_gif_number = existing_gif_count + 1
                        plt.figure(figsize=(15, 10))
                        #plt.plot(self.gp_original.X_time_original, self.gp_original.y_pred_original, label='original gp line')
                        plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'],  color='black', label='original trajectory')
                        plt.savefig(f"image/Patient{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}_original")
                        plt.close()
                        plt.figure(figsize=(15, 10))
                        plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], color='green', linewidth=3, label='corrected trajectory')
                        plt.savefig(f"image/Patient{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}_corrected")
                        plt.close()
                        plt.figure(figsize=(15, 10))
                        plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'],  color='black', label='original trajectory')
                        plt.plot(self.corrected_datas['header'], self.corrected_datas['hip_sagittal'], color='green', linewidth=3, label='corrected trajectory')
                        plt.savefig(f"image/Patient{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}_total")
                        plt.close()
                        # X_pred_gp, y_pred_gp = self.gp.scale(self.corrected_datas['heelstrike'][-1], x_scale, y_scale, self.corrected_datas['header'][-1])
                        # plt.plot(X_pred_gp, y_pred_gp, color='red', label='reference trajectory')

                        #plt.savefig(f"image/Patient{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}")


                    self.scale_histories.append([idx, self.corrected_datas['heelstrike'][-1], self.corrected_datas['header'][-1], detail_scale_hisotries])
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
        existing_gif_count = sum(1 for file in os.listdir("gifs/") if file.startswith("Patient") and file.endswith(".gif"))
        next_gif_number = existing_gif_count + 1
        exportname = f"gifs/Patient{next_gif_number}_Kp{self.Kp}_Ki{self.Ki}_Kd_{self.Kd}"
        duration_rate = 1
        for filename in filenames:
            if filename.endswith(".png"):
                frames.append(imageio.imread(filename))
        imageio.mimsave(f"{exportname}.gif", frames, format='GIF', duration=duration_rate)

        # for filename in set(filenames):
        #     os.remove(filename)
        print('gif saved.')
        if save_data:
            unchange_datas['corrected subject'] = self.corrected_datas.datas

    def minimize_scale(self, idx):
        scale_histories = []
        def objective(params, X, y_actual, heelstrike, interval=0):
            tmp = 0
            scale_x, scale_y = params
            scale_histories.append([scale_x, scale_y])
            X_pred, y_pred = self.gp.scale(heelstrike, scale_x, scale_y, X[-1])
            error = 0
            for x, y in zip(X_pred, y_pred):
                # if x < X[0]:
                #     continue
                distances = np.abs(x - X)
                closest_idxs = np.argsort(distances)[0]
                error += np.square(y - self.corrected_datas['hip_sagittal'][closest_idxs])
            #error = np.mean(error / num)
            error = error
            if tmp % 20 == 0:
                # plt.plot(X_pred, y_pred, color='red')
                # plt.plot(self.corrected_datas['header'], y_actual, color='k')
                # plt.plot(self.original_datas['header'], self.original_datas['hip_sagittal'], color='b', linestyle='--')
                # plt.pause(0.001)
                # plt.cla()
                pass
            tmp+=1
            #print(f"x scale: {scale_x} / y scale: {scale_y} / error: {error}")
            return error
        initial_params = [1.0, 1.0]

        X = self.corrected_datas['header']
        y = self.corrected_datas['hip_sagittal']
        #bounds = [(0.5, 2.0), (0.5, 1.5)]
        #result = minimize(objective, initial_params, args=(X, y, self.corrected_datas['heelstrike'][-1]), method='L-BFGS-B')
        result = minimize(objective, initial_params, args=(X, y, self.corrected_datas['heelstrike'][-1]), method='Powell')
        if result.success:
            optimized_scale_x, optimized_scale_y = result.x
            #print("Optimization successful:", result.message)
            # print("Optimized scales:", optimized_scale_x, optimized_scale_y)
        else:
            print("Optimization failed:", result.message)
            optimized_scale_x, optimized_scale_y = result.x
            #raise ValueError("Optimization failed:", result.message)

        if (idx - (self.start_idx + 62)) % 50 == 0:
            scale_histories = np.array(scale_histories)
            total_frame_len = scale_histories.shape[0]

            scale_x2, scale_y2 = scale_histories[-1]
            x0, _ = self.gp.scale(self.corrected_datas['heelstrike'][-1], scale_x2, scale_y2, X[-1])
            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])
                return line1, line2, line3, title
            
            def update(frame):
                if frame >= total_frame_len:
                    frame2 = total_frame_len - 1
                else:
                    frame2 = frame
                scale_x, scale_y = scale_histories[frame2]
                line1.set_data(self.corrected_datas['header'][:-1], self.corrected_datas['hip_sagittal'][:idx-1])
                x1, y1 = self.gp.scale(self.corrected_datas['heelstrike'][-1], scale_x, scale_y, X[-1])
                line2.set_data(x1, y1)
                if frame >= total_frame_len:
                    #alpha = (frame - frame2) / 10.01   
                    x2, y2 = self.gp.scale(self.corrected_datas['heelstrike'][-1], scale_x, scale_y, X[-1], end=False)
                    control_len = int(len(y2) / 4)
                    line3.set_data(x2[:control_len], y2[:control_len])
                    #line3.set_alpha(alpha)
                title = ax.set_title(f'Scaling process (Total scaling repeat num: {total_frame_len} / now: {frame2+1})')
                return line1, line2, line3, title

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(x0[0]-0.3, x0[-1]-0.3)
            ax.set_ylim(-20, 30)
            line1, = ax.plot([], [], 'k', label='Original Trajectory')
            line2, = ax.plot([], [], 'r', label='Reference Trajectory')
            line3, = ax.plot([], [], 'r--', label='Control Trajectory')
            title = ax.set_title(f'Scaling process (Total scaling repeat num: {total_frame_len} / now: {0})')
            frames = range(0, total_frame_len + 30, 2)
            ax.legend()

            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)
            #writervideo = animation.FFMpegWriter(fps=60)
            ani.save(f'scaling_gifs/scaling process_{idx}.gif', writer='imagemagick', fps=25, dpi=100)
            print("ANIMATION SAVED")
            plt.close()
        return result.x, scale_histories

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




# for idx, heelstrike, header, scale_histories in control.scale_histories:
#     plt.plot(control.original_datas['header'], control.original_datas['hip_sagittal'], label='original subject hip sagittal trajectory')
#     plt.plot(control.corrected_datas['header'][control.start_idx:], control.corrected_datas['hip_sagittal'][control.start_idx:],
#              color='green', linewidth=3, label='corrected subject')

#     for scale_x, scale_y in scale_histories:
#         X_pred_gp, y_pred_gp = control.gp.scale(control.corrected_datas['heelstrike'][-1], scale_x, y_scascale_yle, control.corrected_datas['header'][-1])
#         plt.plot(X_pred_gp, y_pred_gp, color='red', label='reference trajectory') #이게 앞쪽

#     half_len = int(len(y_pred_gp) / 4)
#     X_pred_gp, y_pred_gp = control.gp.scale(control.corrected_datas['heelstrike'][-1], scale_histories[0][0], scale_histories[0][1],
#                                                      control.corrected_datas['header'][-1], end=False) #이게 뒤쪽
#     plt.plot(X_pred_gp, y_pred_gp)

