import numpy as np
from joblib import load
from subject import Subject
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
show_graph = 0

class GP:
    def __init__(self, subject, path='gaussian_process_regressor.joblib'):
        self.subject = subject
        self.data = subject.datas
        self.data_num = len(self.data)
        self.path = path
        self.model = load(self.path)

    def predict(self, X, save_data=True, first_time=False): #polar coordinate으로 predict 뽑는게 디폴트
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        if save_data:
            self.X = X
            self.y_pred = y_pred
            self.sigma = sigma
        if first_time:
            self.X_original = self.X.copy()
            self.X_scalar_original = self.X_scalar.copy()
            self.X_time_original = self.times.copy()
            self.y_pred_original = self.y_pred.copy()
            self.sigma_original = self.sigma.copy()
        return y_pred, sigma

    def _init(self, times, heelstrikes, heelstrike_x, heelstrike_y): #scalar 값으로 predict 계산시 사용
        self.times = times
        HeelStrike =heelstrikes
        self.X_scalar = HeelStrike
        heel_strike_x = heelstrike_x
        heel_strike_y = heelstrike_y
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return self.predict(X, first_time=True)

    def find(self, pos):
        return self.y_pred[pos]

    def find_pred_value_by_heelstrike(self, x_scalar, section): # (HeelStrike) Scalar 값으로 gp 모델의 predict 값 반환
        start_idx, end_idx = self.subject.heel_strike_indices[section:section+2]
        if x_scalar in self.X_scalar[start_idx:end_idx]:
            return self.y_pred[np.where(x_scalar == self.X_scalar[start_idx:end_idx])]
        else:
            closest_indices = np.argsort(np.abs(self.X_scalar - x_scalar))[:2]
            closest_x_values = self.X_scalar[closest_indices]
            closest_y_values = self.y_pred[closest_indices]
            x0, x1 = closest_x_values
            y0, y1 = closest_y_values
            interpolated_value = y0 + (y1 - y0) * (x_scalar - x0) / (x1 - x0)
            return interpolated_value


    def translation(self, delx=0, dely=0, save=True):
        if save:
            self.X_scalar += delx
            self.y_pred -= dely
        else:
            return self.X_scalar + delx, self.y_pred - dely

    def update(self, datas):
        heel_strike_x = datas['heelstrike_x']
        heel_strike_y = datas['heelstrike_y']
        X = np.column_stack((heel_strike_x, heel_strike_y))
        self.X = X
        self.y_pred = self.predict(X)

    def scale(self, heelstrike, scale_x=1.0, scale_y=1.0, x_pos=0.0, end=True, save_data=False, idx=-1):
        #print('scale interval: ', heelstrike - 100, heelstrike)
        data_len = len(self.times)
        if end:
            heelstrike = np.linspace(heelstrike - 100, heelstrike, data_len)  * (2 * np.pi / 100)
            X_time = self.times.copy()
            X_time += x_pos - X_time[-1]
        else:
            #data_len = int(data_len / 10)
            data_len = int(data_len)
            heelstrike = np.linspace(heelstrike, heelstrike + 100, data_len)  * (2 * np.pi / 100)
            X_time = self.times.copy()
            X_time = X_time[:data_len]
            X_time += x_pos - X_time[0]
    
        heelstrike_x = np.cos(heelstrike)
        heelstrike_y = np.sin(heelstrike)
        X = np.column_stack((heelstrike_x, heelstrike_y))
        y_pred, _ = self.predict(X, save_data=False)
        
        if x_pos != 0.0:
            X_scalar_gp = (X_time - x_pos) * scale_x + x_pos
        else:
            X_scalar_gp = X_time * scale_x
        y_pred_gp = y_pred * scale_y

        # global show_graph
        # if show_graph % 2 == 0:
        #     plt.plot(X_scalar_gp, y_pred_gp, color='red', label='scaled')
        #     plt.scatter(X_time, y_pred, color='k', label='original')
        #     plt.legend()
        #     plt.pause(0.001)
        #     plt.cla()
        # show_graph += 1

        if save_data == False:
            if idx != -1:
                return X_scalar_gp[:idx], y_pred_gp[:idx]
            else:
                return X_scalar_gp, y_pred_gp
        else:
            self.times = X
            self.y_pred = y_pred_gp
            return X_scalar_gp, y_pred_gp
        

if __name__ == "__main__":
    subject = Subject(6, cut=True)
    gp = GP(subject)
    gp.scale(scale_x=1.1)