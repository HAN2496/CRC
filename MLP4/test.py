import os
import pandas as pd
import numpy as np

SUBJECTS = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 30] # All subjects
#SUBJECTS = [6, 8]

base_path = os.path.dirname(__file__)  # Gets the directory of the current script
file_path = os.path.join(base_path, 'EPIC', 'subject_info.csv')
subject_info = pd.read_csv(file_path)

class Subject:
    def __init__(self, number):
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

                data.append({
                    'name': filename.split('.')[0],
                    'header': header,
                    'hip_sagittal': hip_sagittal,
                    'heelstrike_x': heelstrike_x,
                    'heelstrike_y': heelstrike_y,
                    'heelstrike': heelstrike,
                    'heelstrike_speed': heelstrike_speed,
                    'foot_Accel_X': foot_Accel_X,
                    'foot_Accel_Y': foot_Accel_Y,
                    'foot_Accel_Z': foot_Accel_Z
                })
        self.datas = np.array(data)

    def calc_torque(self):
        for data in self.datas:
            hip_sagittal = data['hip_sagittal']
            vel = np.diff(hip_sagittal, prepend=0)
            acc = np.diff(vel, prepend=0)
        pass
    def divide_by_section(self): #
        self.heel_strike_indices = []
        for entry in self.datas:
            heel_strike_indices = self.find_zero_indices(entry['heel_strike'], 0)
            self.heel_strike_indices.append(heel_strike_indices)

    @staticmethod
    def find_zero_indices(values_list, target_value):
        return [index for index, value in enumerate(values_list) if value == target_value]

    def extract_by_heelstrike_range(self, start=0, end=90, index_pos=4):
        start, end = int(start), int(end)
        random_test_num = np.random.randint(0, len(self.datas)) #데이터 중 선택
        entry = self.datas[random_test_num]

        heel_strike_indices = self.heel_strike_indices[random_test_num]
    
        start_idx = heel_strike_indices[index_pos]
        end_idx = heel_strike_indices[index_pos] - 1


        if start < 0:
            num, val = index_pos-1, 100 + start
        else:
            num, val = index_pos, start

        start_idx_within_stride = np.argmin(np.abs(entry['heel_strike'][heel_strike_indices[num]:heel_strike_indices[num+1]] - val))
        start_idx = heel_strike_indices[num] + start_idx_within_stride

        end_idx_within_stride = np.argmin(np.abs(entry['heel_strike'][heel_strike_indices[4]:heel_strike_indices[5]] - end))
        end_idx = heel_strike_indices[4] + end_idx_within_stride

        selected_data = {
            'hip_sagittal': entry['hip_sagittal'][start_idx:end_idx+1],
            'heel_strike_x': entry['heel_strike_x'][start_idx:end_idx+1],
            'heel_strike_y': entry['heel_strike_y'][start_idx:end_idx+1],
            'heel_strike': entry['heel_strike'][start_idx:end_idx+1]
        }
        return selected_data
    

a = Subject(6)