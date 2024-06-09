import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from datasets import Datasets
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, spline_filter
from pykalman import KalmanFilter




class Subject:
    def __init__(self):

        self.name = "Patient"
        self.info = {
            "Weight": np.array([1.0], dtype=np.float32),
            "thigh": np.array([1.0], dtype=np.float32),
            "calf": np.array([1.0], dtype=np.float32)
        }

        file_path = "CP child gait data/dicp/DiCP3a_x.csv"
        file = pd.read_csv(file_path)

        hip_sagittal = file['hip_sagittal'].values
        hip_sagittal = savgol_filter(hip_sagittal, window_length=30, polyorder=3)  # 스무딩
        hip_sagittal_speed = np.diff(hip_sagittal, prepend=0) / 0.005
        hip_sagittal_acc = np.diff(hip_sagittal_speed, prepend=0) / 0.005

        heelstrike = file['heelstrike'].values
        heelstrike_radians = (heelstrike / 100.0) * 2 * np.pi
        heelstrike_x = np.cos(heelstrike_radians)
        heelstrike_y = np.sin(heelstrike_radians)
 
        header = file['header'].values
        torque = self.calc_torque(hip_sagittal)

        sections = file['section'].values
        data = {
            'section': sections,
            'header': header,
            'hip_sagittal': hip_sagittal,
            'hip_sagittal_speed': hip_sagittal_speed,
            'hip_sagittal_acc': hip_sagittal_acc,
            'heelstrike_x': heelstrike_x,
            'heelstrike_y': heelstrike_y,
            'heelstrike': heelstrike,
            'torque': torque
        }
        self.datas = data
        self.find_start_index()
    
        #self.datas = np.array([data])
        self.datas = Datasets(self.datas)
        self.spring = 1
        self.damper = 1


    def move(self, torque_input):
        m = self.info['Weight']
        l_thigh = self.info['thigh']
        l_calf = self.info['calf']
        m_thigh = m * 0.28
        m_calf = m * 0.08
        m_toe = m * 0.05
        I_thigh = 2/3 * m_thigh * l_thigh
        I_calf = 2/3 * m_calf * l_calf + m_calf * (l_thigh+l_calf) **2
        I_toe = m_toe * (l_thigh + l_calf) ** 2
        I_total = I_thigh + I_calf + I_toe
        alpha = torque_input / I_total
        return alpha


    def calc_torque(self, hip_sagittal, extract=False):
        m = self.info['Weight']
        l_thigh = self.info['thigh']
        l_calf = self.info['calf']
        m_thigh = m * 0.28
        m_calf = m * 0.08
        m_toe = m * 0.05
        I_thigh = 2/3 * m_thigh * l_thigh
        I_calf = 2/3 * m_calf * l_calf + m_calf * (l_thigh+l_calf) **2
        I_toe = m_toe * (l_thigh + l_calf) ** 2
        I_total = I_thigh + I_calf + I_toe
        pos = savgol_filter(hip_sagittal, window_length=30, polyorder=3)  # 스무딩
        #pos = pd.Series(hip_sagittal).rolling(window=20, min_periods=5, center=True).mean().values
        #pos = gaussian_filter1d(hip_sagittal, sigma=3)
        vel = np.diff(pos, prepend=pos[0]) / 0.005
        acc = np.diff(vel, prepend=vel[0]) / 0.005
        if extract:
            return pos, vel, acc
        else:
            torque = I_total * acc# + self.damper * vel + self.spring * pos
            return torque

    def find_start_index(self):
        self.start_indices = []
        heelstrike = self.datas['heelstrike']
        start_indices = self.find_zero_indices(heelstrike, 0)
        self.start_indices.append(start_indices)

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return np.array([index for index, value in enumerate(values_list) if value == target_value])


    def get_random_test_num(self):
        return np.random.randint(0, len(self.datas)) #데이터 중 선택

    def plot(self, num=0, show=True):
        plt.figure(figsize=(10, 6))
        #plt.plot(self.datas['header'], self.datas['hip_sagittal'], label='hip sagittal')
        #plt.plot(self.datas['header'], self.datas['hip_sagittal_speed'], label='hip sagittal')
        plt.plot(self.datas['header'][5:], self.datas['hip_sagittal_acc'][5:], color='black', label='hip sagittal acc')
        torque = self.datas['torque'] * 0.5
        alpha = self.move(torque)
        plt.plot(self.datas['header'][5:], alpha[5:], color='green')

        if show:
            plt.title(f'Original dataset for subject')
            plt.legend()
            #ax2.legend(loc='upper right')
            plt.show()

if __name__ == "__main__":
    subject = Subject()
    print(subject.datas['header'])
    subject.plot()