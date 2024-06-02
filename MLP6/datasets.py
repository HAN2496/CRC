import numpy as np
from matplotlib import pyplot as plt

class Datasets:
    def __init__(self, datas=None):
        if datas is not None:
            self.datas = datas
        else:
            self.datas = {
                'section': [],
                'header': [],
                'hip_sagittal': [],
                'hip_sagittal_speed': [],
                'hip_sagittal_acc': [],
                'heelstrike': [],
                'heelstrike_x': [],
                'heelstrike_y': [],
                'torque': []
            }
        self.additional_datas = {}

    def __str__(self):
        summary_dict = {key: values for key, values in self.datas.items()}
        additional_summary = {key: len(values) for key, values in self.additional_datas.items()}
        return f"{summary_dict} \n Additional: {additional_summary}"

    def __getitem__(self, key):
        if key in self.datas:
            return self.datas[key]
        elif key in self.additional_datas:
            return self.additional_datas[key]
        else:
            raise KeyError(f"Key {key} not found in datas or additional_datas")

    def append(self, key, data):
        target_dict = self.datas if key in self.datas else self.additional_datas
        if key not in target_dict:
            target_dict[key] = []
        if isinstance(data, list):
            target_dict[key].extend(data)
        elif isinstance(data, np.ndarray):
            if len(target_dict[key]) == 0:
                target_dict[key] = data.tolist()
            else:
                target_dict[key].extend(data.tolist())
        else:
            target_dict[key].append(data)
    
    def appends(self, data):
        for key, value in data.items():
            self.append(key, value)

    def index(self, key, idx):
        data = self.__getitem__(key)
        if isinstance(idx, list):
            return [data[i] for i in idx]
        else:
            return data[idx]

    def indexs(self, idx):
        if idx == -1:
            indexed_data = {key: self.__getitem__(key)[:] for key in self.datas}
        elif isinstance(idx, list):
            indexed_data = {key: [self.__getitem__(key)[i] for i in idx] for key in self.datas}
        else:
            indexed_data = {key: self.__getitem__(key)[idx] for key in self.datas}
        return indexed_data

class Test:
    def __init__(self):
        self.data = Datasets()
        self.data.appends({'test': 0})
        self.data.append('test', 1)
        self.data.index('test', 0)
        print(self.data)
        print(self.data['test'][0])

#test = Test()