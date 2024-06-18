import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import c3d
from scipy.signal import find_peaks

BASE_PATH = Path(__file__).parent
file_path = BASE_PATH / "DiCP3a.c3d"
datasets = c3d.Reader(open(file_path, 'rb')) #read bytes type
frames = list(datasets.read_frames())
labels = datasets.point_labels
marker_name = "RHipAngles"
marker_data = []

with open(file_path, 'rb') as f:
    reader = c3d.Reader(f)
    labels = reader.point_labels
    labels = np.char.strip(labels.astype(str))
    marker_index = np.where(labels == marker_name)[0][0]
    for frame_no, points, analog in reader.read_frames():
        marker_position = points[marker_index]
        marker_data.append(marker_position)
marker_data = np.array(marker_data)

def remove_edge_zeros(arr):
    start = np.argmax(arr != 0)
    end = len(arr) - np.argmax(arr[::-1] != 0)
    return arr[start:end]

x_data = marker_data[:, 0]
cutted_data = remove_edge_zeros(x_data)


max_real, min_real = 24.67578, -14.8726
max_cutted, min_cutted = max(cutted_data), min(cutted_data)

a, b = max_real, abs(min_real)
d, c = max_cutted, min_cutted
k = (-c + d) / (a + b)
e = d - a * k

cutted_data = cutted_data - e

"""
100Hz -> 200 Hz
"""
current_time = np.linspace(0, len(cutted_data) - 1, num=len(cutted_data))
new_time = np.linspace(0, len(cutted_data) - 1, num=2 * len(cutted_data) - 1)
interpolated_data = np.interp(new_time, current_time, cutted_data)


ts = new_time / 200
hip_sagittal = interpolated_data


points = np.array([114, 276, 440])
points_interval = np.mean(np.array([points[idx+1] - points[idx] for idx in range(len(points) - 1)]))

plt.plot(range(len(interpolated_data)), interpolated_data)
plt.scatter(points, interpolated_data[points])
plt.show()
coordinates = np.array([(index, hip_sagittal[index]) for index in points])

heelstrikes = []
heelstrike = 0
heelstrikes.extend(np.linspace(100 - 100 * points[0] / points_interval, 100 , int(points[0])))

for idx in range(len(points) - 1):
    num = points[idx+1] - points[idx]
    heelstrike = np.linspace(0, 100, num)
    heelstrikes.extend(heelstrike)

print(f"len : {len(heelstrikes), len(hip_sagittal)}")
length = len(hip_sagittal) - len(heelstrikes)
heelstrike = np.array([i * length / points_interval for i in range(length)])
heelstrikes.extend(heelstrike)

headers = []
header = 0
for _ in heelstrikes:
    headers.append(header)
    header += 0.005

sections = []
section = 0
tmp = 0
for heelstrike in heelstrikes:
    if heelstrike == 0:
        if tmp == 0:
            tmp+= 1
        else:
            section += 1
    sections.append(section)
total_data = np.column_stack((headers, sections, heelstrikes, interpolated_data))
# df = pd.DataFrame(total_data, columns=['header', 'section', 'heelstrike', 'hip_sagittal'])
# df.to_csv("DiCP2.csv", index=False)




plt.plot(range(len(hip_sagittal)), hip_sagittal)
ax2 = plt.twinx()
ax2.plot(range(len(heelstrikes)), heelstrikes)
plt.show()










