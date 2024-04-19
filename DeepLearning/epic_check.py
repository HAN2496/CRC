import numpy as np
import pandas as pd
import c3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from tsai.all import *
import numpy as np
from keras.models import load_model

def to_polar_coordinates(value):
    theta = (value / 100) * 2 * np.pi
    xs = np.cos(theta)
    ys = np.sin(theta)
    return np.column_stack((xs, ys))

def cubic_interpolate(x, y, interval=0.001):
    x = np.array(x)
    y = np.array(y)
    a = x[1] - x[0]
    b = y[1] - y[0]
    p = - 2 * b / a ** 3
    q = 3 * b / a ** 2
    interpolate_x = np.arange(x[0], x[1], interval)
    interpolate_y = np.array([p * (i - x[0]) ** 3 + q * (i - x[0]) ** 2 for i in interpolate_x])
    return np.column_stack((interpolate_x, interpolate_y + y[0]))

path='datasets/epic/AB06/10_09_18/levelground/gon/'
path_phase='datasets/epic/AB06/10_09_18/levelground/gcRight/'
data1 = 'levelground_ccw_normal_01_01.csv'
data2 = 'levelground_ccw_normal_01_01.csv'
data_header = pd.read_csv(path+data1)[['Header']].values[::5]
data_hip = pd.read_csv(path+data1)[['hip_sagittal']].values[::5]
data_phase = pd.read_csv(path_phase+data1)[['HeelStrike']].values

"""
start_index = 5200 / 5
end_index1 = 5600 / 5
end_index2 = 6360 / 5


data_header_idx1 = data_header[start_index:end_index1]
data_hip_idx1 = data_hip[start_index:end_index1]
cubic1 = cubic_interpolate([data_header[start_index], data_header[end_index1]], [data_hip[start_index], data_hip[end_index1]])

data_header_idx2 = data_header[end_index1:end_index2]
data_hip_idx2 = data_hip[end_index1:end_index2]
cubic2 = cubic_interpolate([data_header[end_index1], data_header[end_index2]], [data_hip[end_index1], data_hip[end_index2]])


data_header_idx3 = data_header[start_index:end_index2]
data_hip_idx3 = data_hip[start_index:end_index2]
cubic3 = np.concatenate((cubic1, cubic2))

width = 2
# 전체 플롯에 대한 설정
#plt.figure(figsize=(10, 8))

# 첫 번째 플롯
plt.subplot2grid((2, 2), (0, 0))
plt.plot(data_header_idx1, data_hip_idx1, label='half cycle', linewidth=3)
plt.plot(cubic1[:, 0], cubic1[:, 1], label='cubic interpolate', linewidth=3)
plt.ylabel('Hip joint angle (deg)')
plt.legend()

# 두 번째 플롯
plt.subplot2grid((2, 2), (0, 1))
plt.plot(data_header_idx2, data_hip_idx2, label='half cycle', linewidth=3)
plt.plot(cubic2[:, 0], cubic2[:, 1], label='cubic interpolate', linewidth=3)
plt.legend()

# 세 번째 플롯 (가로로 긴 플롯)
plt.subplot2grid((2, 2), (1, 0), colspan=2)
plt.plot(data_header_idx3, data_hip_idx3, label='one cycle', linewidth=3)
plt.plot(cubic3[:, 0], cubic3[:, 1], label='cubic interpolate', linewidth=3)
plt.xlabel('time (sec)')
plt.ylabel('Hip joint angle (deg)')
plt.legend()

# 전체 제목 추가
plt.suptitle('Comparison of Real Hip Joint Angles and Interpolation', y=0.95)

# 전체 레이아웃을 조정하여 레이블과 제목이 겹치지 않도록 합니다.
#plt.tight_layout(rect=[0, 0, 1, 0.95])
"""

"""
Estimation part 부분
"""

fig, ax1 = plt.subplots()
ax1.plot(data_header, data_hip, color='blue')
ax1.set_ylabel("Hip joint angle (Deg)", color='blue')

ax2 = ax1.twinx()
ax2.plot(data_header, data_phase, color='red')
ax2.set_ylabel("Gait phase (0 ~ 100)", color='red')

plt.title('Hip joint angle and Gait phase (Scalar)')
# 그래프 표시
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(data_header, data_hip, color='blue')
ax1.set_ylabel("Hip joint angle (Deg)", color='blue')

ax2 = ax1.twinx()
ax2.plot(data_header, data_phase, color='red')
ax2.set_ylabel("Gait phase (rad)", color='red')
plt.title('Hip joint angle and Gait phase (Polar coordinates)')
plt.show()