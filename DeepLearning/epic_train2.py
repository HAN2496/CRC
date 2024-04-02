import numpy as np
import pandas as pd
import tensorflow as tf
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from tsai.all import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


shift_interval = +10
prefix = "name"

path = 'datasets/epic'
datasets = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]
    if i==0:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_02_01.csv"
    else:
        total_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_02.csv"
    mat_file = pd.read_csv(total_path).iloc[:, [0, 4]]
    data_len = len(mat_file)
    mat_file['Header'] = np.linspace(0, 1, data_len)
    datasets.append(mat_file.to_numpy())

    if i == 1:
        print("*"*50)
        print("Check data")
        print(mat_file)
        print("*"*50)

#datasets = pd.concat(datasets, axis=0)

print(datasets)


k = 5  # 미래 예측을 위한 시간 단계

# 입력과 라벨 데이터를 준비하는 함수
def create_dataset(data, k=5):
    X, y = [], []
    for i in range(len(data) - k):
        X.append(data[i:(i + k), :])  # 현재 시점에서 k 스텝 전까지의 데이터
        y.append(data[i + k, :])  # k 스텝 후의 데이터
    return np.array(X), np.array(y)

# 데이터셋 생성
datasets_X, datasets_y = [], []
for data in datasets:
    X, y = create_dataset(data, k)
    datasets_X.append(X)
    datasets_y.append(y)

# 제너레이터 함수 정의
def data_generator():
    for X, y in zip(datasets_X, datasets_y):
        yield X, y

# TensorFlow 데이터셋 생성
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, k, 2), dtype=tf.float64),  # 입력 시퀀스
        tf.TensorSpec(shape=(2,), dtype=tf.float64)  # 출력 값
    )
)

# 패딩 추가 및 배치 처리
dataset = dataset.padded_batch(
    batch_size=2,
    padded_shapes=([None, k, 2], [2]),  # 입력 및 출력 패딩 설정
    padding_values=(0.0, 0.0)  # 입력 및 출력 패딩 값
)

# 모델 정의
model = Sequential([
    Masking(mask_value=0.0, input_shape=(k, 2)),  # 패딩된 값 무시
    LSTM(32),  # LSTM 레이어
    Dense(2)  # 출력 레이어, 2개의 값을 예측
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(dataset, epochs=10)