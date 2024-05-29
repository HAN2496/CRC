import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
#SUBJECTS = [6, 8]

base_path = os.path.dirname(__file__)  # Gets the directory of the current script
file_path = os.path.join(base_path, 'EPIC', 'subject_info.csv')
subject_info = pd.read_csv(file_path)
"""
class Datasets:
    def __init__(self):
        self.data = []

    def append(self, header, hip_sgittal, heelstrike):
        self.data.append({
            'name': filename.split('.')[0],
            'header': header,
            'hip_sagittal': hip_sagittal,
            'heelstrike_x': heelstrike_x,
            'heelstrike_y': heelstrike_y,
            'heelstrike': heelstrike,
            'heelstrike_speed': heelstrike_speed,
            'torque': torque,
            'foot_Accel_X': foot_Accel_X,
            'foot_Accel_Y': foot_Accel_Y,
            'foot_Accel_Z': foot_Accel_Z
        })
"""

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
        self.find_walking_start_point()
        if cut:
            self.datas = self.extract_by_heelstrike_range()

    def calc_torque(self, hip_sagittal):
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
        self.smoothed_data = savgol_filter(hip_sagittal, window_length=5, polyorder=2)  # 스무딩
        vel = savgol_filter(hip_sagittal, window_length=5, polyorder=2, deriv=1, delta=1.0)  # 속도 (1차 미분)
        acc = savgol_filter(hip_sagittal, window_length=5, polyorder=2, deriv=2, delta=1.0)  # 가속도 (2차 미분)
        #vel = np.diff(hip_sagittal, prepend=0) * 200
        #acc = np.diff(vel, prepend=0) * 200
        torque = I_total * acc
        return torque

    def find_walking_start_point(self): #
        self.heel_strike_indices = []
        for entry in self.datas:
            heel_strike_indices = self.find_zero_indices(entry['heelstrike'], 0)
            self.heel_strike_indices.append(heel_strike_indices)

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return [index for index, value in enumerate(values_list) if value == target_value]

    def extract_by_heelstrike_range(self, start=0, end=100, index_pos=4):
        start, end = int(start), int(end)
        random_test_num = self.used_data_idx #데이터 중 선택
        entry = self.datas[random_test_num]

        heel_strike_indices = self.heel_strike_indices[random_test_num]
    
        start_idx = heel_strike_indices[index_pos]
        end_idx = heel_strike_indices[index_pos] - 1


        if start < 0:
            num, val = index_pos-1, 100 + start
        else:
            num, val = index_pos, start

        start_idx_within_stride = np.argmin(np.abs(entry['heelstrike'][heel_strike_indices[num]:heel_strike_indices[num+1]] - val))
        start_idx = heel_strike_indices[num] + start_idx_within_stride

        end_idx_within_stride = np.argmin(np.abs(entry['heelstrike'][heel_strike_indices[4]:heel_strike_indices[5]] - end))
        end_idx = heel_strike_indices[4] + end_idx_within_stride

        selected_data = {
            'header': entry['header'][start_idx:end_idx+1],
            'hip_sagittal': entry['hip_sagittal'][start_idx:end_idx+1],
            'heelstrike_x': entry['heelstrike_x'][start_idx:end_idx+1],
            'heelstrike_y': entry['heelstrike_y'][start_idx:end_idx+1],
            'heelstrike': entry['heelstrike'][start_idx:end_idx+1],
            'torque': entry['torque'][start_idx:end_idx+1]
        }
        return selected_data

    def get_random_test_num(self):
        return np.random.randint(0, len(self.datas)) #데이터 중 선택

    def plot(self, num=0, show=True):
        plt.figure(figsize=(10, 6))
        #plt.plot(self.datas[self.used_data_idx]['header'], self.datas[self.used_data_idx]['hip_sagittal'], label='hip sagittal')
        plt.plot(self.datas['header'], self.datas['hip_sagittal'], label='hip sagittal')
        #plt.plot(self.cutted_datas['header'], self.cutted_datas['hip_sagittal'])
        ax2 = plt.gca().twinx()
        #ax2.plot(self.datas[self.used_data_idx]['header'], self.datas[self.used_data_idx]['torque'], label='torque', color='red')
        ax2.plot(self.datas['header'], self.datas['torque'], label='torque', color='red')

        #for y_val in self.heel_strike_indices[self.used_data_idx]:
        #    plt.axvline(self.datas[self.used_data_idx]['header'][y_val])

        if show:
            plt.title(f'Original dataset for subject')
            plt.legend()
            ax2.legend(loc='upper right')
            plt.show()

if __name__ == "__main__":
    subject = Subject(6, cut=True)
    subject.plot()