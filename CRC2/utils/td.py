import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.subject import Subject
from utils.datasets import Datasets
from configs.config_datasets import TD_SUBJECTS, TD_FOLDER

class TD(Subject):
    def __init__(self, number, choose_one_dataset=True, extract_walking=True):
        super().__init__(number, TD_FOLDER, TD_SUBJECTS, extract_walking=extract_walking)
        if number not in TD_SUBJECTS:
            raise ValueError(f"Subject number should be in {TD_SUBJECTS}")
        if choose_one_dataset:
            self.extract_one_datasets()
        if extract_walking:
            self.extract_walking_phases()

    def load_subject_info(self):
        file_path = self.base_path / TD_FOLDER / 'subject_info.csv'
        self.subject_info = pd.read_csv(file_path)
        self.info = self.subject_info[self.subject_info["Subject"] == f"AB{self.number}"]
        self.inertia = self.calc_inertia()

    def load_data(self):
        heelstrike_path = self.base_path / TD_FOLDER / f"AB{self.number}" / 'gcRight'
        gon_path = self.base_path / TD_FOLDER / f"AB{self.number}" / 'gon'
        imu_path = self.base_path / TD_FOLDER / f"AB{self.number}" / 'imu'
        filenames = []
        data_len = []
        for filename in os.listdir(heelstrike_path):
            if filename.endswith('.csv'):
                filenames.append(filename)
                df_heelstrike = pd.read_csv(heelstrike_path / filename)
                df_gon = pd.read_csv(gon_path / filename)
                df_imu = pd.read_csv(imu_path / filename)
                self.process_heelstrike(df_heelstrike)
                self.process_hip_sagittal(df_gon[::5])
                self.process_imu(df_imu)
                data_len.append(len(df_heelstrike))

        self.datas.append('file_names', np.array(filenames))
        self.datas.append('length of each file', np.array(data_len))

    def extract_specific_datasets(self, number):
        pass
    def extract_normal_walking_datasets(self):
        filtered_file_idx = [idx for idx, file_name in enumerate(self.datas['file_names']) if "normal" in file_name]
        filtered_file_len = len(filtered_file_idx)
        return filtered_file_idx[np.random.randint(0, filtered_file_len)]

    def extract_one_datasets(self, extract_normal_phase=False):
        datas = Datasets()
        file_num = len(self.datas['file_names'])
        if extract_normal_phase:
            self.random_data_num = self.extract_normal_walking_datasets()
        else:
            self.random_data_num = np.random.randint(0, file_num)
        data_len = np.array(self.datas['length of each file'][self.random_data_num])
        file_name = self.datas['file_names'][self.random_data_num]
        print(f"Used data: {file_name}")

        file_start_idx = sum(np.array(self.datas['length of each file'][:self.random_data_num]))
        file_end_idx = file_start_idx + data_len
        filtered_indices = list(range(file_start_idx, file_end_idx))

        filtered_data = self.datas.indexs(filtered_indices)
        datas.appends(filtered_data)
        self.datas = datas
        self.datas.append('file_names', file_name)
        self.datas.append('length of each file', np.array([data_len]))
        print(f"Data len: {len(self.datas)}")

    def extract_walking_phases(self, section_start=4, interval=3):
        file_start_idx = 0
        datas = Datasets()
        file_names = []
        data_len = []
        for file_name, each_datasets_len in zip(self.datas['file_names'], self.datas['length of each file']):
            idxs = list(range(file_start_idx, file_start_idx + each_datasets_len))
            file_names.append(file_name)
            entry = self.datas.indexs(idxs)
            filtered_indices = np.array([idx for idx, section in enumerate(entry['section']) if section_start <= section < section_start + interval]) + file_start_idx
            data_len.append(len(filtered_indices))
            if len(filtered_indices) == 0:
                ValueError("No data found for the specified section range.")
            else:
                filtered_data = self.datas.indexs(filtered_indices.tolist())
            datas.appends(filtered_data)
            file_start_idx += each_datasets_len

        self.datas = datas
        self.datas.redefine_indexs()
        self.datas.append('file_names', file_names)
        self.datas.append('length of each file', data_len)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        if len(self.datas['length of each file']) != 1:
           self.extract_one_datasets()
        print(len(self.datas['header']))

    
        ax.plot(self.datas['header'], self.datas['hip_sagittal'])
        ax.plot(self.datas['header'], self.datas['hip_sagittal_a'])
        print(self.datas['file_names'])
        filename = self.datas['file_names'][0][:-4]
        # if self.extract_walking:
        #    ax.text(0.5, 0.96, 'Extract walking phases only', fontsize=10, ha='center', va='center', transform=ax.transAxes)
        plt.title(f'Used TD datasets: {filename}')
        plt.xlabel('time (sec)')
        plt.ylabel('hip sagittal angle (deg)')
        plt.show()

if __name__ == "__main__":
    td_data = TD(6, choose_one_dataset=True, extract_walking=True)
    td_data.plot()