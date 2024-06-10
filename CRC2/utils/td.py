import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.subject import Subject
from utils.datasets import Datasets
from configs.config_datasets import TD_SUBJECTS, TD_FOLDER

class TD(Subject):
    def __init__(self, number, extract_walking=True):
        super().__init__(number, TD_FOLDER, TD_SUBJECTS, extract_walking=extract_walking)
        if number not in TD_SUBJECTS:
            raise ValueError(f"Subject number should be in {TD_SUBJECTS}")
        if extract_walking:
            self.cut_data()

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

    def cut_data(self, section_start=4, interval=3):
        file_num = len(self.datas['file_names'])
        self.random_data_num = np.random.randint(0, file_num)
        data_len = self.datas['length of each file'][self.random_data_num]
        print(f"Used data: {self.datas['file_names'][self.random_data_num]}")
        file_start_idx = sum(self.datas['length of each file'][:self.random_data_num])
        file_end_idx = file_start_idx + data_len
        entry = self.datas[file_start_idx:file_end_idx]

        filtered_indices = [idx for idx, section in enumerate(entry['section']) if section_start <= section < section_start + interval]
        if len(filtered_indices) == 0:
            ValueError("No data found for the specified section range.")
        else:
            filtered_data = self.datas.indexs(filtered_indices)

        self.datas = Datasets(filtered_data)
        self.datas.redifine_indexs()
        print(f"Data len: {len(self.datas)}")

    def plot(self):
        plt.plot(self.datas['header'], self.datas['hip_sagittal'])

        plt.title('TD datasets')
        plt.show()

if __name__ == "__main__":
    td_data = TD(6)
    td_data.plot()