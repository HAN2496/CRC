import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from utils.datasets import Datasets

class Subject:
    def __init__(self, number, data_folder, subjects_list, choose_one_dataset=True, extract_walking=True):
        self.extract_walking = extract_walking
        self.choose_one_dataset = choose_one_dataset
        if number not in subjects_list:
            raise ValueError(f"Subject number should be in {subjects_list}")
        if str(data_folder) == "EPIC":
            self.number = str(number).zfill(2)
        else:
            self.number = number
        self.base_path = Path(__file__).parent.parent / "datasets"
        self.data_folder = data_folder
        self.datas = Datasets()
        self.load_subject_info()
        self.load_data()
        self.find_start_indices()

    def load_subject_info(self):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()

    def process_heelstrike(self, df):
        radians = (df['HeelStrike'].values / 100.0) * 2 * np.pi
        self.datas.append('heelstrike_x', np.cos(radians))
        self.datas.append('heelstrike_y', np.sin(radians))
        self.datas.append('heelstrike',  df['HeelStrike'].values)
        sections, section = [], 0
        for heelstrike in df['HeelStrike'].values:
            if heelstrike == 0:
                section += 1
            sections.append(section)
        self.datas.append('section', sections)
        self.datas.append('header', df['Header'].values)

    def process_hip_sagittal(self, df):
        hip_sagittal = df['hip_sagittal'].values
        pos, vel, acc = self.calc_kinematics(hip_sagittal)
        torque = self.inertia * acc
        self.datas.append('hip_sagittal', pos)
        self.datas.append('hip_sagittal_v', vel)
        self.datas.append('hip_sagittal_a',  acc)
        self.datas.append('torque',  torque)

    def process_imu(self, df):
        self.datas.append('foot_Accel_X',  df['foot_Accel_X'].values)
        self.datas.append('foot_Accel_Y',  df['foot_Accel_Y'].values)
        self.datas.append('foot_Accel_Z',  df['foot_Accel_Z'].values)

    def find_start_indices(self):
        self.start_indices = self.find_indices(self.datas['heelstrike'], 0)

    @staticmethod
    def find_indices(values_list, target_value):
        return np.array([index for index, value in enumerate(values_list) if value == target_value])

    def move(self, torque_input):
        acc = torque_input / self.inertia
        return acc

    def calc_kinematics(self, pos, smooth=True):
        if smooth:
            pos = savgol_filter(pos, window_length=30, polyorder=3)  # Smoothing
        vel = np.diff(pos, prepend=pos[0]) / 0.005
        acc = np.diff(vel, prepend=vel[0]) / 0.005
        return pos, vel, acc

    def calc_torque(self, hip_sagittal):
        _, _, acc = self.calc_kinematics(hip_sagittal, smooth=False)
        return self.inertia * acc

    def calc_inertia(self):
        m = self.info['Weight'].values[0]
        l_thigh = self.info['thigh'].values[0]
        l_calf = self.info['calf'].values[0]
        m_thigh = m * 0.28
        m_calf = m * 0.08
        m_toe = m * 0.05
        I_thigh = 2/3 * m_thigh * l_thigh
        I_calf = 2/3 * m_calf * l_calf + m_calf * (l_thigh + l_calf) ** 2
        I_toe = m_toe * (l_thigh + l_calf) ** 2
        return I_thigh + I_calf + I_toe

    def plot(self, data, title='Data Visualization'):
        plt.plot(data)
        plt.title(title)
        plt.show()

