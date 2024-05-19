import numpy as np
from data_loader import load_data
import matplotlib.pyplot as plt
from joblib import dump

def find_all_indexes(lst, value):
    return [index for index, current in enumerate(lst) if current == value]

smooth = False
hips, heel_rads, heels = load_data(smooth=smooth)

data_len = len(hips)
print(f"Total data length: {data_len}")

hip_starts, hip_mids, hip_ends = [], [], []
heel_starts, heel_mids, heel_ends = [], [], []
heel_start_rads, heel_mid_rads, heel_end_rads = [], [], []
heel_0_idxs, heel_100_idxs = [], []


for i in range(data_len):
    current_hips = hips[i]
    current_heels = heels[i]
    current_heel_rads = heel_rads[i]
    heel_0_idx = find_all_indexes(heels[i], 0)
    heel_0_idxs.append(heel_0_idx)
    heel_100_idx = find_all_indexes(heels[i], 100)
    heel_100_idxs.append(heel_100_idx)

    tmp2 = 3
    hip_starts.append(current_hips[heel_0_idx[0]:heel_0_idx[tmp2]])
    heel_starts.append(current_heels[heel_0_idx[0]:heel_0_idx[tmp2]])

    tmp = -2
    hip_mids.append(current_hips[heel_0_idx[tmp2]:heel_100_idx[tmp]])
    heel_mids.append(current_heels[heel_0_idx[tmp2]:heel_100_idx[tmp]])
    heel_mid_rads.append(current_heel_rads[heel_0_idx[2]:heel_100_idx[tmp], :])

    hip_ends.append(current_hips[heel_100_idx[tmp]+1:heel_100_idx[-1]])
    heel_ends.append(current_heels[heel_100_idx[tmp]+1:heel_100_idx[-1]])


#check_num = np.random.randint(0, data_len)
#print(f"check num: {check_num}")
for check_num in range(data_len):
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)
    ax1 = plt.gca()
    ax1.plot(range(len(hips[check_num])), hips[check_num], label='Hips', color='b')
    ax1.set_ylabel('Hips', color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(heels[check_num])), heels[check_num], label='Heels', color='r')
    ax2.set_ylabel('Heels', color='r')
    ax1.set_title('Hips and Heels Comparison')  # 제목 추가

    plt.subplot(2, 2, 2)
    ax1 = plt.gca()
    ax1.plot(range(len(hip_starts[check_num])), hip_starts[check_num], label='Hip Starts', color='b')
    ax1.set_ylabel('Hip Starts', color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(heel_starts[check_num])), heel_starts[check_num], label='Heel Starts', color='r')
    ax2.set_ylabel('Heel Starts', color='r')
    ax1.set_title('Hip Starts and Heel Starts Comparison')  # 제목 추가

    plt.subplot(2, 2, 3)
    ax1 = plt.gca()
    ax1.plot(range(len(hip_mids[check_num])), hip_mids[check_num], label='Hip Mids', color='b')
    ax1.set_ylabel('Hip Mids', color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(heel_mids[check_num])), heel_mids[check_num], label='Heel Mids', color='r')
    ax2.set_ylabel('Heel Mids', color='r')
    ax1.set_title('Hip Mids and Heel Mids Comparison')  # 제목 추가

    plt.subplot(2, 2, 4)
    ax1 = plt.gca()
    ax1.plot(range(len(hip_ends[check_num])), hip_ends[check_num], label='Hip Ends', color='b')
    ax1.set_ylabel('Hip Ends', color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(heel_ends[check_num])), heel_ends[check_num], label='Heel Ends', color='r')
    ax2.set_ylabel('Heel Ends', color='r')
    ax1.set_title('Hip Ends and Heel Ends Comparison')  # 제목 추가

    plt.savefig(f'figure/{check_num}_image.png')