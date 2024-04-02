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
from tensorflow.keras.layers import LSTM, Dense
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
    datasets.append(mat_file)

    if i == 1:
        print("*"*50)
        print("Check data")
        print(mat_file)
        print("*"*50)

datasets = pd.concat(datasets, axis=0)

print(datasets)

# 예측하려는 미래의 시간 단계 (k)
k = 5

# 예시 DataFrame 데이터셋 생성
datasets = [
    pd.DataFrame(np.random.rand(15000, 2)),
    pd.DataFrame(np.random.rand(20000, 2)),
    # 추가 데이터셋 ...
]

def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        # 마지막 가능한 시작 인덱스 찾기: 이를 통해 시퀀스가 k 단계 뒤를 예측할 수 있음
        end_ix = i + n_steps + k
        # 시퀀스 끝에 도달했는지 확인
        if end_ix > len(sequences):
            break
        # 입력과 출력 파트 분할
        seq_x, seq_y = sequences[i:end_ix-k], sequences[end_ix-k:end_ix]
        X.append(seq_x)
        y.append(seq_y[-1])  # 마지막 타임스텝만 예측 대상으로 사용
    return np.array(X), np.array(y)

# 제너레이터 함수 정의
def data_generator():
    for data in datasets:
        # DataFrame을 넘파이 배열로 변환
        sequence = data.to_numpy()
        # 입력과 타겟 시퀀스 분할
        X, y = split_sequences(sequence, n_steps=20)  # 예: 20개의 타임스텝을 사용하여 시퀀스를 분할
        for i in range(len(X)):
            yield X[i], y[i]

# TensorFlow 데이터셋 생성
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64),  # 입력 시퀀스
        tf.TensorSpec(shape=(2,), dtype=tf.float64)  # 타겟 값
    )
)

# 패딩 추가 및 배치 처리
dataset = dataset.padded_batch(
    batch_size=32,
    padded_shapes=([None, 2], [2]),  # 입력 및 타겟 패딩
    padding_values=(0.0, 0.0)  # 입력 및 타겟 패딩 값
)

# 모델 정의
model = Sequential([
    Masking(mask_value=0.0, input_shape=(None, 2)),  # 패딩된 값 무시
    LSTM(32),  # LSTM 레이어
    Dense(2)  # 출력 레이어, 타겟이 (2,) 형태이므로 2개의 유닛
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(dataset, epochs=10)
