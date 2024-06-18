import numpy as np
from joblib import load, dump
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

from utils.td import TD
from configs.config_reference import *

class Reference:
    def __init__(self, fit_mode=False, X=None, y=None, path='models/gaussian_process_regressor.joblib'):
        if fit_mode:
            if X is None or y is None:
                raise ValueError("X and y must be provided if fit_mode is True")
            self.fit(X, y)
        self.model = load(path)

    def fit(self, heelstrike, y):
        self.heelstrike = np.array(heelstrike)
        self.heelstrike_x, self.heelstrike_y = self.scalar_to_rad(self.heelstrike)
        X = np.column_stack((self.heelstrike_x, self.heelstrike_y))
        kernel = ExpSineSquared(length_scale=LENGTH_SCALE, periodicity=PERIODICITY
                        , length_scale_bounds=LENGTH_SCALE_BOUNDS,
                        periodicity_bounds=PERIODICITY_BOUNDS
                        ) + WhiteKernel(noise_level=NOISE_LEVEL)
        reference = GaussianProcessRegressor(kernel=kernel)

        reference.fit(X, y)
        self.model = reference
        self.y_pred, self.sigma = self.predict(X)
        path = 'models/gaussian_process_regressor.joblib'
        dump(reference, path)
        print(f"Gaussian Process model saved at {path}")

    def predict(self, X):
        X = np.array(X)
        if X.shape[1] != 2:
            raise ValueError("")
        y_pred, sigma = self.model.predict(X, return_std=True)
        return y_pred, sigma

    def scalar_to_rad(self, heelstrike):
        radians = (heelstrike / 100.0) * 2 * np.pi
        return np.cos(radians), np.sin(radians)

    def update(self, times, heelstrike):
        self.times = np.array(times)
        self.heelstrike = np.array(heelstrike)
        self.heelstrike_x, self.heelstrike_y = self.scalar_to_rad(self.heelstrike)
        X = np.column_stack((self.heelstrike_x, self.heelstrike_y))
        self.y_pred, self.sigma = self.predict(X)


    def move(self, x, y, delx, dely):
        return x + delx, y - dely

    def scale(self, heelstrike, time, scale_x, scale_y, scale_from_end=True):
        closest_index = min(range(len(self.heelstrike)), key=lambda i: abs(self.heelstrike[i] - heelstrike))
        y_pred_sorted = np.roll(self.y_pred, -closest_index)
        times = self.times.copy()
        if scale_from_end:
            times =  times + (time - times[-1])
        else:
            times = times + (time - times[0])
        
        times = (times - time) * scale_x + time
        y_pred = y_pred_sorted * scale_y


        return times, y_pred

    def plot(self):
        plt.plot(range(len(self.y_pred)), self.y_pred)
        plt.show()

if __name__ == "__main__":
    reference = Reference()
    td = TD(number=6, choose_one_dataset=False, extract_walking=False)
    td.extract_one_datasets(True)
    datas = td.datas
    reference.update(datas['header'], datas['heelstrike'])
    time, y_pred = reference.scale(40, 1.0, 1.2, 1.5, 1.0)
    plt.plot(time, y_pred, color='red', linestyle='--')
    plt.show()
    reference.plot()
