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
                'hip_sagittal_v': [],
                'hip_sagittal_a': [],
                'heelstrike': [],
                'heelstrike_x': [],
                'heelstrike_y': [],
                'torque': []
            }
        self.additional_datas = {}

    def __len__(self):
        return len(self.datas['header']) 

    def __str__(self):
        summary_dict = {key: len(values) for key, values in self.datas.items()}
        additional_summary = {key: len(values) for key, values in self.additional_datas.items()}
        return f"{'*' * 80}\nData lengths of datasets \n {summary_dict} \n Additional: {additional_summary}\n{'*' * 80}"

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.datas:
                return self.datas[key]
            elif key in self.additional_datas:
                return self.additional_datas[key]
            else:
                raise KeyError(f"Key {key} not found in datas or additional_datas")
        elif isinstance(key, int) or isinstance(key, slice):
            return {k: v[key] for k, v in self.datas.items() if len(v) > max(key.start if isinstance(key, slice) else 0, 0)}
        else:
            raise TypeError("Invalid key type. Must be str, int, or slice.")
    
    def redefine_indexs(self):
        self.datas['section'] = np.array(self.datas['section']) - self.datas['section'][0]
    
    def append(self, key, data):
        target_dict = self.datas if key in self.datas else self.additional_datas
        if key not in target_dict:
            target_dict[key] = []
        if isinstance(data, list):
            if len(target_dict[key]) == 0:
                target_dict[key] = data
            else:
                target_dict[key].extend(data)
        elif isinstance(data, np.ndarray):
            if len(target_dict[key]) == 0:
                target_dict[key] = list(data)
            else:
                target_dict[key].extend(data.tolist())
        else:
            target_dict[key].append(data)

    def appends(self, data):
        for key, value in data.items():
            self.append(key, value)

    def find_by_hip_sagittal(self, hip_sagittal, section):
        idxs = np.where(np.array(self.datas['section']) == section)[0]
        if len(idxs) == 0:
            raise ValueError

        hip_sagittal_values = np.array([self.datas['hip_sagittal'][idx] for idx in idxs])
        
        if hip_sagittal in hip_sagittal_values:
            return idxs[np.where(hip_sagittal_values == hip_sagittal)[0][0]]
        
        distances = np.abs(hip_sagittal_values - hip_sagittal)
        closest_idxs = np.argsort(distances)[0]
        return idxs[closest_idxs]

    def sections(self, section, index=False):
        indices = [i for i, sec in enumerate(self.datas['section']) if sec == section]
        if not indices:
            return None
        if index:
            return indices
        filtered_data = {}
        for key in self.datas:
            if all(i < len(self.datas[key]) for i in indices):
                filtered_data[key] = [self.datas[key][i] for i in indices if key != "section"]
            else:
                filtered_data[key] = None
        return filtered_data

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
        self.data.appends({'section': [1, 1, 2, 2], 'header': [10, 20, 30, 40], 'hip_sagittal': [100, 200, 300, 400],
                           'total_sections': [0, 1, 2]})
        print(self.data)
        print("="*50)
        print(self.data['header'])
        print(self.data.sections(1))
        print(self.data.sections(2, index=True))
        print(self.data.sections(3))

if __name__ == "__main":
    test = Test()