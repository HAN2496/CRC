import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from utils.subject import Subject
from configs.config_datasets import CP_SUBJECTS, CP_FOLDER


class CP(Subject):
    def __init__(self, number):
        super().__init__(number, CP_FOLDER, CP_SUBJECTS, choose_one_dataset=True, extract_walking=False)


    def load_subject_info(self):
        file_path = self.base_path / CP_FOLDER / 'subject_info.csv'
        self.subject_info = pd.read_csv(file_path)
        self.info = self.subject_info[self.subject_info["Subject"] == self.number]
        self.inertia = self.calc_inertia()

    def load_data(self):
        file_path = self.base_path / CP_FOLDER / "dicp" / f"DiCP{self.number}.csv"
        file = pd.read_csv(file_path)

        self.process_hip_sagittal(file)
        self.process_heelstrike(file)
    
        self.datas.append('file_names', np.array([f"DiCP{self.number}"]))
        self.datas.append('length of each file', np.array([len(file)]))
    
    def plot(self):
        plt.plot(self.datas['header'], self.datas['hip_sagittal'])
        plt.show()

if __name__ == "__main__":
    cp = CP(2)
    cp.plot()