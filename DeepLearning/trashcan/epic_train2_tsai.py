import numpy as np
import os
import matplotlib.pyplot as plt
from tsai.all import *
from fastai.data.all import *
from sklearn.model_selection import train_test_split
from tsai.all import TSDatasets, TSClassifier, Learner, accuracy, InceptionTime
"""
Step1: csv 데이터 로드
 - levelground_ccw_normal_01_02.csv 데이터를 datasets에 모두 넣어줌.
 - 총 데이터 수는 22개
 - 
"""

shift_interval = +10
prefix = "name"

path = 'datasets/epic'

X = []
y = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    if i==0:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_02_01.csv"
    else:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_02.csv"
    mat_file = pd.read_csv(total_path).iloc[::5, 4].values
    data_len = len(mat_file)
    gait_phase = np.linspace(0, 1, data_len)
    X.append(mat_file)
    y.append(gait_phase)

max_len = max(len(x) for x in X)

y_padded = []
for ys in y:
    if len(ys) < max_len:
        padding = np.array([0 for _ in range(max_len - len(ys))])
        ys = np.concatenate((ys, padding))
    y_padded.append(ys)

def pad_sequence(seq, max_len):
    pad_len = max_len - len(seq)
    return np.pad(seq, (0, pad_len), 'constant', constant_values=(0, 0))

X_padded = np.array([pad_sequence(x, max_len) for x in X])

tsds = TSDatasets(X_padded, y_padded, tfms=None)

masks = np.array([[False if val == 0 else True for val in seq] for seq in X_padded])
splits = np.random.RandomState(seed=42).permutation(len(X_padded))
cut = int(len(X_padded) * 0.8)  # 80%는 학습, 20%는 검증용
train_split, valid_split = splits[:cut], splits[cut:]

X_train, X_valid, y_train, y_valid = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)


tsds = TSDatasets(X_padded, y_padded, splits=(train_split, valid_split), tfms=[None, None])

# 모델 선택
n_features = tsds.train[0][0].shape[1]
n_classes = len(np.unique(y_padded))
model = InceptionTime(n_features, n_classes, nf=64)

# Learner 객체 생성 및 학습
learn = Learner(tsds.dataloaders(bs=64), model, metrics=accuracy)

# 모델 학습 (예: 10 에포크)
learn.fit_one_cycle(10)

"""

"""

"""
for i in datasets:
    plt.plot(i[:, 0], i[:, 1])
plt.show()
"""




"""
Step2: 데이터 전처리
 - 데이터 패딩 처리
 - train, valid, test 데이터 분리

max_len_datasets = max([len(x) for x in datasets])
X = np.array([np.pad(x[:-shift_interval], ((0, max_len_datasets - len(x) + shift_interval), (0, 0)), 'constant', constant_values=0) for x in datasets])
max_len_y = max([len(yi) for yi in y])
y = np.array([np.pad(yi, (0, max_len_y - len(yi)), 'constant', constant_values=0) for yi in y])





X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
splits = get_splits(y_train, valid_size=0.2)  # 훈련 데이터를 다시 훈련/검증 세트로 분할

Step3:데이터 학습


# tsai에서 데이터셋을 생성

tfms = [None, None]  # 라벨 변환을 위한 변환
dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits, inplace=True)

# DataLoader 생성
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64)

# 모델 선택 및 학습
# InceptionTime 모델 사용 (회귀 문제에 적합하도록 조정 필요)
model = InceptionTime(dls.vars, 1)  # 회귀 문제의 경우 출력 크기를 1로 설정 (단일 값 예측을 위함)
learn = Learner(dls, model, loss_func=MSELossFlat())  # 회귀 문제에 적합한 손실 함수 사용

# 모델 학습 시작
learn.fit_one_cycle(5)  # 5 에포크 동안 학습
"""