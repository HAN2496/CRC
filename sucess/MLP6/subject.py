import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from datasets import Datasets
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, spline_filter
from pykalman import KalmanFilter

SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
#SUBJECTS = [6, 8]

base_path = os.path.dirname(__file__) 
file_path = os.path.join(base_path, 'EPIC', 'subject_info.csv')
subject_info = pd.read_csv(file_path)

class Subject:
    def __init__(self, number, cut=False):
        self.cut = cut
        if number not in SUBJECTS:
            raise ValueError("Wrong number")

        self.name = str(number).zfill(2)
        self.info = subject_info[subject_info["Subject"] == f"AB{self.name}"]
        heelstrike_path = f"EPIC/AB{self.name}/gcRight"
        gon_path = f"EPIC/AB{self.name}/gon"
        imu_path = f"EPIC/AB{self.name}/imu"
        heelstrike_files = os.listdir(heelstrike_path)
        data = []

        for filename in heelstrike_files:
            if filename.endswith('.csv') and "ccw_normal" in filename:
                df_heelstrike = pd.read_csv(os.path.join(heelstrike_path, filename))
                df_gon = pd.read_csv(os.path.join(gon_path, filename))
                df_imu = pd.read_csv(os.path.join(imu_path, filename))

                hip_sagittal = df_gon['hip_sagittal'][::5].values
                hip_sagittal_speed = np.diff(hip_sagittal, prepend=0)
                hip_sagittal_acc = np.diff(hip_sagittal_speed, prepend=0)
                heelstrike = df_heelstrike['HeelStrike'].values
                header = df_heelstrike['Header'].values
                heelstrike_speed = np.diff(heelstrike, prepend=0)
                foot_Accel_X = df_imu['foot_Accel_X'].values
                foot_Accel_Y = df_imu['foot_Accel_Y'].values
                foot_Accel_Z = df_imu['foot_Accel_Z'].values

                heelstrike_radians = (heelstrike / 100.0) * 2 * np.pi
                heelstrike_x = np.cos(heelstrike_radians)
                heelstrike_y = np.sin(heelstrike_radians)

                torque = self.calc_torque(hip_sagittal)
                data.append({
                    'name': filename.split('.')[0],
                    'header': header,
                    'hip_sagittal': hip_sagittal,
                    'hip_sagittal_speed': hip_sagittal_speed,
                    'hip_sagittal_acc': hip_sagittal_acc,
                    'heelstrike_x': heelstrike_x,
                    'heelstrike_y': heelstrike_y,
                    'heelstrike': heelstrike,
                    'heelstrike_speed': heelstrike_speed,
                    'torque': torque,
                    'foot_Accel_X': foot_Accel_X,
                    'foot_Accel_Y': foot_Accel_Y,
                    'foot_Accel_Z': foot_Accel_Z
                })
        self.datas = np.array(data)
        self.used_data_idx = self.get_random_test_num()
        self.used_data_idx = 1
        self.find_start_index()
        if cut:
            self.datas, section = self.cut_data(interval=7)
        pos, vel, acc = self.calc_torque(self.datas['hip_sagittal'], extract=True)
        self.datas['hip_sagittal'] = pos
        self.datas['hip_sagittal_speed'] = vel
        self.datas['hip_sagittal_acc'] = acc
        self.datas = Datasets(self.datas)
        self.datas.append("total_sections", [i for i in range(section)])

    def move(self, torque_input):
        m = self.info['Weight'].values
        l_thigh = self.info['thigh'].values
        l_calf = self.info['calf'].values
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
        m = self.info['Weight'].values
        l_thigh = self.info['thigh'].values
        l_calf = self.info['calf'].values
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
        #vel = np.diff(pos, prepend=pos[0]) / 0.005
        #acc = np.diff(vel, prepend=vel[0]) / 0.005
        vel = np.gradient(pos, 0.005)
        acc = np.gradient(vel, 0.005)
        if extract:
            return pos, vel, acc
        else:
            torque = I_total * acc
            return torque

    def find_start_index(self): #
        self.start_indices = []
        for data in self.datas:
            start_indices = self.find_zero_indices(data['heelstrike'], 0)
            self.start_indices.append(start_indices)

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return np.array([index for index, value in enumerate(values_list) if value == target_value])

    def cut_data(self, start=0, end=100, index_pos=4, interval=2):
        start, end = int(start), int(end)
        random_test_num = self.used_data_idx #데이터 중 선택
        entry = self.datas[random_test_num]

        start_indices = self.start_indices[random_test_num]

        start_idx = start_indices[index_pos]
        end_idx = start_indices[index_pos + interval]
        self.start_indices = self.find_zero_indices(entry['heelstrike'][start_idx:end_idx], 0)

        sections, section = [], 0
        for heelstrike in entry['heelstrike'][start_idx:end_idx]:
            if heelstrike == 0:
                section += 1
            sections.append(section)
        sections = np.array(sections) - 1

        selected_data = {
            'section': sections,
            'header': entry['header'][start_idx:end_idx],
            'hip_sagittal': entry['hip_sagittal'][start_idx:end_idx],
            'hip_sagittal_speed': entry['hip_sagittal_speed'][start_idx:end_idx],
            'hip_sagittal_acc': entry['hip_sagittal_acc'][start_idx:end_idx],
            'heelstrike_x': entry['heelstrike_x'][start_idx:end_idx],
            'heelstrike_y': entry['heelstrike_y'][start_idx:end_idx],
            'heelstrike': entry['heelstrike'][start_idx:end_idx],
            'torque': entry['torque'][start_idx:end_idx]
        }
        return selected_data, section

    def get_random_test_num(self):
        return np.random.randint(0, len(self.datas)) #데이터 중 선택

    def plot(self, num=0, show=True):
        plt.figure(figsize=(10, 6))
        #plt.plot(self.datas['header'], self.datas['hip_sagittal'], label='hip sagittal')
        #plt.plot(self.datas['header'], self.datas['hip_sagittal_speed'], label='hip sagittal')
        plt.plot(self.datas['header'], self.datas['hip_sagittal_acc'], label='hip sagittal acc')
        
        #plt.plot(self.cutted_datas['header'], self.cutted_datas['hip_sagittal'])
        ax2 = plt.gca().twinx()
        #ax2.plot(self.datas[self.used_data_idx]['header'], self.datas[self.used_data_idx]['torque'], label='torque', color='red')
        ax2.plot(self.datas['header'], self.datas['torque'], label='torque', color='red')

        for y_val in self.start_indices:
            plt.axvline(self.datas['header'][y_val], linestyle='--')

        if show:
            plt.title(f'Original dataset for subject')
            plt.legend()
            ax2.legend(loc='upper right')
            plt.show()

if __name__ == "__main__":
    subject = Subject(6, cut=True)
    subject.plot()