import numpy as np
from joblib import load

class GP:
    def __init__(self, data_num, path='gaussian_process_regressor.joblib'):
        self.data_num = data_num
        self.path = path
        self.model = load(self.path)
        self.predict_by_scalar(data_num)
        self.X_original = self.X.copy()
        self.X_scalar_original = self.X_scalar.copy()
        self.y_pred_original = self.y_pred.copy()

    def predict(self, X): #polar coordinate으로 predict 뽑는게 디폴트
        X = np.array(X)
        y_pred, sigma = self.model.predict(X, return_std=True)
        self.X = X
        self.y_pred = y_pred
        self.sigma = sigma

        return y_pred, sigma

    def predict_by_scalar(self, num, start=0, end=100): #scalar 값으로 predict 계산시 사용
        HeelStrike = np.linspace(start, end, num)
        self.X_scalar = HeelStrike
        heel_strike_radians = (HeelStrike / 100.0) * 2 * np.pi
        heel_strike_x = np.cos(heel_strike_radians)
        heel_strike_y = np.sin(heel_strike_radians)
        X = np.column_stack((heel_strike_x, heel_strike_y))
        return self.predict(X)

    def find_pred_value_by_scalar(self, x_scalar): # (HeelStrike) Scalar 값으로 gp 모델의 predict 값 반환
        if x_scalar in self.X_scalar:
            return self.y_pred[np.where(x_scalar == self.X_scalar)]
        else:
            closest_indices = np.argsort(np.abs(self.X_scalar - x_scalar))[:2]
            closest_x_values = self.X_scalar[closest_indices]
            closest_y_values = self.y_pred[closest_indices]
            x0, x1 = closest_x_values
            y0, y1 = closest_y_values
            interpolated_value = y0 + (y1 - y0) * (x_scalar - x0) / (x1 - x0)
            return interpolated_value

    def translation(self, delx=0, dely=0):
        self.X_scalar += delx
        self.y_pred -= dely

    def scale(self, scale_x=1, scale_y=1, x_pos=0, y_pos=0, save_data=False):
        X_scalar_gp = self.X_scalar.copy() - x_pos
        y_pred_gp = self.y_pred.copy() - y_pos

        X_scalar_gp *= scale_x
        y_pred_gp *= scale_y

        X_scalar_gp += x_pos
        y_pred_gp += y_pos

        if save_data == False:
            return X_scalar_gp, y_pred_gp
        else:
            self.X_scalar = X_scalar_gp
            self.y_pred = y_pred_gp