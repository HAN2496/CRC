import numpy as np
import os
import matplotlib.pyplot as plt
from tsai.all import *
from fastai.data.all import *
from sklearn.model_selection import train_test_split
from tsai.all import TSDatasets, TSClassifier, Learner, accuracy, InceptionTime
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def to_polar_coordinates(value):
    theta = (value / 100) * 2 * np.pi
    xs = np.cos(theta)
    ys = np.sin(theta)
    return np.column_stack((xs, ys))



shift_interval = +10
window_length = 1000
stride_num = 100
prefix = "test"

path = 'datasets/epic'

X = []
y = []
data_num = [i for i in range(25) if i not in [16, 20, 23]]
for i in data_num:
    patient_path = f"{path}/AB{str(i+6).zfill(2)}"
    folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
    specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"]

    if i==0:
        input_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_02_01.csv"
        target_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gcRight/levelground_ccw_normal_02_01.csv"

    else:
        input_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gon/levelground_ccw_normal_01_02.csv"
        target_path = f"{patient_path}/{specific_folder_path[0]}/levelground/gcRight/levelground_ccw_normal_01_02.csv"

    input = pd.read_csv(input_path).iloc[::5, 4].values
    target = pd.read_csv(target_path).iloc[:, 1].values
    target = to_polar_coordinates(target)
    target_scalar = target
    sw = SlidingWindow(window_length, stride=stride_num, get_y=None)
    X_window, _ = sw(input)
    y_window, _ = sw(target)

    X.extend(X_window)
    y.extend(y_window)

X = np.array(X)
y = np.array(y)
y_scalar = y
y = y.reshape(y.shape[0], -1)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X shape: {X.shape[1]}, y shape: {y.shape[1]}")
print(f"Number of samples in X: {len(X)}")
print(f"Number of targets: {len(y)}")


splits = get_splits(y, valid_size=0.2, stratify=False, random_state=23, shuffle=True)
tfms  = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, num_workers=0)


# 모델 초기화
model = InceptionTime(X.shape[1], y.shape[1])

# Learner 객체 생성
learn = Learner(dls, model, metrics=rmse)

learn.load("test2")
learn.model.eval()

index_num = np.random.randint(564)
example_input_array = X[index_num].reshape(1, X.shape[1], X.shape[2])

example_input_tensor = torch.tensor(example_input_array, dtype=torch.float)

# 예측 실행
with torch.no_grad():
    prediction = learn.model(example_input_tensor)

print(prediction)
actual_value = y[index_num]

# 예측 결과와 실제 값 비교
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(prediction[0].numpy(), label='Predicted')
plt.plot(actual_value, label='Actual')
plt.title('Prediction vs Actual')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

print(actual_value.shape)

def xy_to_scaled_value(x, y):
    """
    (x, y) 좌표를 받아서 해당 좌표가 나타내는 각도를 0에서 100 사이의 값으로 변환합니다.
    """
    # arctan2를 사용하여 theta 계산
    theta = np.arctan2(y, x)
    
    # theta가 음수인 경우, 2pi를 더해 양수로 만듦
    if theta < 0:
        theta += 2 * np.pi
    
    # theta를 0에서 100으로 스케일링
    scaled_value = (theta / (2 * np.pi)) * 100
    
    return scaled_value

actual_value_scalar = []
prediction_scalar = []
tmp = prediction[0].numpy()
for i in range(len(actual_value)):
    if i % 2 == 0:
        actual_value_scalar.append(xy_to_scaled_value(actual_value[i], actual_value[i+1]))
        prediction_scalar.append(xy_to_scaled_value(tmp[i], tmp[i+1]))

actual_value_scalar = np.array(actual_value_scalar)
prediction_scalar = np.array(prediction_scalar)
print(actual_value_scalar.shape)
print(prediction_scalar.shape)

plt.subplot(212)
plt.plot(prediction_scalar, label='Predicted')
plt.plot(actual_value_scalar, label='Actual')
plt.title('Prediction vs Actual in scalar(0 ~ 100)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
