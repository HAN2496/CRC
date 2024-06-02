import numpy as np
from joblib import load
from subject import Subject
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class GP:
    def __init__(self, subject, path='gaussian_process_regressor.joblib'):
        self.subject = subject
        self.data = subject.datas
        self.data_num = len(self.data)
        self.path = path
        self.model = load(self.path)
        self.predict_by_heelstrike()
        self.X_original = self.X.copy()
        self.X_scalar_original = self.X_scalar.copy()
        self.y_pred_original = self.y_pred.copy()

    def predict(self, X, save_data=True): #polar coordinate으로 predict 뽑는게 디폴트
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        if save_data:
            self.X = X
            self.y_pred = y_pred
            self.sigma = sigma
        return y_pred, sigma

    def predict_by_heelstrike(self, start=0, end=100): #scalar 값으로 predict 계산시 사용
        HeelStrike = self.data['heelstrike']
        self.X_scalar = HeelStrike
        heel_strike_x = self.data['heelstrike_x']
        heel_strike_y = self.data['heelstrike_y']
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return self.predict(X)

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


    def scale(self, scale_x=1.0, scale_y=1.0, x_pos=0, y_pos=0, save_data=False, idx=-1):
        X_scalar_gp = self.X_scalar - x_pos
        y_pred_gp = self.y_pred - y_pos

        X_scalar_gp *= scale_x
        y_pred_gp *= scale_y

        X_scalar_gp += x_pos
        y_pred_gp += y_pos

        if scale_x != 1.0:
            cutted_datas = []
            interpolated_y = []
            for section in self.data['total_sections']:
                idxs = self.data.sections(section, index=True)
                cutted_datas.append([X_scalar_gp[idxs], y_pred_gp[idxs], self.X_scalar[idxs]])
            for x, y, original in cutted_datas:
                interpolator = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
                y_pred = interpolator(original)
                interpolated_y.extend(y_pred)
            y_pred_gp = np.array(interpolated_y)
            if idx != -1:
                y_diff = y_pos - y_pred_gp[idx-1]
                y_pred_gp = y_pred_gp - y_diff

        if save_data == False:
            if idx != -1:
                return X_scalar_gp[:idx], y_pred_gp[:idx]
            else:
                return X_scalar_gp, y_pred_gp
        else:
            #self.X_scalar = X_scalar_gp
            self.y_pred = y_pred_gp
        

if __name__ == "__main__":
    subject = Subject(6, cut=True)
    gp = GP(subject)
    gp.scale(scale_x=1.1)