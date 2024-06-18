import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import c3d
from scipy.signal import find_peaks

file_path = "CP child gait data/dicp/DiCP3a.c3d"
marker_name = "RHipAngles"
datasets = c3d.Reader(open(file_path, 'rb')) #read bytes type
frames = list(datasets.read_frames())
labels = datasets.point_labels
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

#df = pd.DataFrame(marker_data[:, :3], columns=['X', 'Y', 'Z'])

x_data = marker_data[:, 0]
cutted_data = x_data[158+15:546]

max_real, min_real = 24.67578, -14.8726
max_cutted, min_cutted = max(cutted_data), min(cutted_data)

a, b = max_real, abs(min_real)
d, c = max_cutted, min_cutted
k = (-c + d) / (a + b)
e = d - a * k

cutted_data = cutted_data - e

current_time = np.linspace(0, len(cutted_data) - 1, num=len(cutted_data))

new_time = np.linspace(0, len(cutted_data) - 1, num=2 * len(cutted_data) - 1)

interpolated_data = np.interp(new_time, current_time, cutted_data)








ts = new_time / 200
hip_sagittal = interpolated_data
points = np.array([25, 240, 465, 687]) - 25 # 215, 225, 222
coordinates = np.array([(index, hip_sagittal[index]) for index in points])

heelstrikes = []
for idx in range(len(points) - 1):
    num = points[idx+1] - points[idx]
    heelstrike = np.linspace(0, 100, num)
    heelstrikes.extend(heelstrike)

print(f"len : {len(heelstrikes), len(hip_sagittal)}")
length = len(hip_sagittal) - len(heelstrikes)
heelstrike = np.array([i * length / 220 for i in range(length)])
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
# df.to_csv("CP child gait data/dicp/DiCP3a_x.csv", index=False)




plt.plot(range(len(hip_sagittal)), hip_sagittal)
plt.plot(range(len(heelstrikes)), heelstrikes)
plt.show()





















gon_path = "EPIC/AB06/gon/levelground_ccw_normal_02_01.csv"
heelstrike_path = "EPIC/AB06/gcRight/levelground_ccw_normal_02_01.csv"

df_gon = pd.read_csv(gon_path)['hip_sagittal'][::5].values
df_heelstrike = pd.read_csv(heelstrike_path)['HeelStrike'].values

fig, ax1 = plt.subplots()

ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Hip Sagittal Angle', color='tab:blue')
ax1.plot(range(len(df_gon)), df_gon, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Heel Strike', color='tab:red')
ax2.plot(range(len(df_heelstrike)), df_heelstrike, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
# plt.show()