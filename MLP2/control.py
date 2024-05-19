import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.gaussian_process.kernels import ExpSineSquared
from data_loader import load_data
from utils import find_top_n_elements_with_indexes

smooth = True

def find_all_indexes(lst, value):
    return [index for index, current in enumerate(lst) if current == value]

hips, _, heels = load_data(smooth=smooth)

data_len = len(hips)
print(f"Total data length: {data_len}")

hip_starts, hip_mids, hip_ends = [], [], []
heel_starts, heel_mids, heel_ends = [], [], []
heel_0_idxs = []

for i in range(data_len):
    current_hips = hips[i]
    current_heels = heels[i]
    heel_0_idx = find_all_indexes(heels[i], 0)
    heel_0_idxs.append(heel_0_idx)

    hip_starts.append(current_hips[heel_0_idx[0]:heel_0_idx[1]])
    heel_starts.append(current_heels[heel_0_idx[0]:heel_0_idx[1]])

    hip_mids.append(current_hips[heel_0_idx[1]:heel_0_idx[-1]])
    heel_mids.append(current_heels[heel_0_idx[1]:heel_0_idx[-1]])

    hip_ends.append(current_hips[heel_0_idx[-1]:-1])
    heel_ends.append(current_heels[heel_0_idx[-1]:-1])


results = []
for idx1, datas in enumerate([hip_starts, hip_mids]):
    for idx, data in enumerate(datas):
        if idx1 == 0:
            name = "starts"
        elif idx1 == 1:
            name = "mids"
        
        # Initialize stats[name] as a dictionary before using it
        stats = {}
        stats[name] = {}

        heel_0_idx_mid_num = len(heel_0_idxs[idx])-2
        
        stats[name]['mean'] = np.mean(data)  # 평균
        stats[name]['std_dev'] = np.std(data)  # 표준편차
        stats[name]['variance'] = np.var(data)  # 분산

        # 주기, 진폭
        peaks, _ = find_peaks(data)
        troughs, _ = find_peaks(-data)

        periods = np.diff(peaks)
        amplitudes = [data[peaks[i]] - data[troughs[i]] for i in range(min(len(peaks), len(troughs)))][1:]
        if len(periods) < len(amplitudes):
            amplitudes = amplitudes[1:]
        else:
            pass

        if not smooth and len(amplitudes) > heel_0_idx_mid_num:
            amplitudes, amplitudes_idx = find_top_n_elements_with_indexes(amplitudes, heel_0_idx_mid_num)
            periods = [periods[idx] for idx in amplitudes_idx]

        stats[name]['period'] = periods
        stats[name]['amplitude'] = amplitudes

        average_period = np.mean(periods)
        average_amplitude = np.mean(amplitudes)
        stats[name]['period_mean'] = average_period
        stats[name]['amplitude_mean'] = average_amplitude

        results.append(stats)


fig, axs = plt.subplots(5, 1, figsize=(10, 10))
titles = ['Mean', 'Standard Deviation', 'Variance', 'Period', 'Amplitude Mean']
means = []
std_devs = []
variances = []
periods = []
amplitude_means = []
for result in results:
    if 'mids' in result:
        means.append(result['mids']['mean'])
        std_devs.append(result['mids']['std_dev'])
        variances.append(result['mids']['variance'])
        periods.append(result['mids']['period_mean'])  # Assuming you want the mean of the periods
        amplitude_means.append(result['mids']['amplitude_mean'])

# Plotting each statistic
axs[0].plot(means, marker='o', linestyle='-')
axs[0].set_title('Mean of Starts')
axs[0].set_ylabel('Mean')

axs[1].plot(std_devs, marker='o', linestyle='-')
axs[1].set_title('Standard Deviation of Starts')
axs[1].set_ylabel('Standard Deviation')

axs[2].plot(variances, marker='o', linestyle='-')
axs[2].set_title('Variance of Starts')
axs[2].set_ylabel('Variance')

axs[3].plot(periods, marker='o', linestyle='-')
axs[3].set_title('Average Period of Starts')
axs[3].set_ylabel('Period')

axs[4].plot(amplitude_means, marker='o', linestyle='-')
axs[4].set_title('Average Amplitude of Starts')
axs[4].set_ylabel('Amplitude')

# Set common labels and layout
for ax, title in zip(axs, titles):
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.show()

















# Initialize accumulators for each statistic
total_means = []
total_std_devs = []
total_variances = []
total_period_means = []
total_amplitude_means = []

# Iterate through the results to collect 'starts' segment data
for result in results:
    if 'starts' in result:
        total_means.append(result['starts']['mean'])
        total_std_devs.append(result['starts']['std_dev'])
        total_variances.append(result['starts']['variance'])
        total_period_means.append(result['starts']['period_mean'])
        total_amplitude_means.append(result['starts']['amplitude_mean'])

# Compute the overall averages for each statistic if there is any data collected
overall_mean = np.mean(total_means) if total_means else 0
overall_std_dev = np.mean(total_std_devs) if total_std_devs else 0
overall_variance = np.mean(total_variances) if total_variances else 0
overall_period_mean = np.mean(total_period_means) if total_period_means else 0
overall_amplitude_mean = np.mean(total_amplitude_means) if total_amplitude_means else 0

# Print the overall averages
print(f"Overall Mean: {overall_mean}")
print(f"Overall Standard Deviation: {overall_std_dev}")
print(f"Overall Variance: {overall_variance}")
print(f"Overall Period Mean: {overall_period_mean}")
print(f"Overall Amplitude Mean: {overall_amplitude_mean}")


















check_num = 0
plt.subplot(2, 2, 1)
ax1 = plt.gca()
ax1.plot(range(len(hips[check_num])), hips[check_num], label='Hips', color='b')
ax1.set_ylabel('Hips', color='b')
ax2 = ax1.twinx()
ax2.plot(range(len(heels[check_num])), heels[check_num], label='Heels', color='r')
ax2.set_ylabel('Heels', color='r')
ax1.set_title('Hips and Heels Comparison')

plt.subplot(2, 2, 2)
ax1 = plt.gca()
ax1.plot(range(len(hip_starts[check_num])), hip_starts[check_num], label='Hip Starts', color='b')
ax1.set_ylabel('Hip Starts', color='b')
ax2 = ax1.twinx()
ax2.plot(range(len(heel_starts[check_num])), heel_starts[check_num], label='Heel Starts', color='r')
ax2.set_ylabel('Heel Starts', color='r')
ax1.set_title('Hip Starts and Heel Starts Comparison')

plt.subplot(2, 2, 3)
ax1 = plt.gca()
ax1.plot(range(len(hip_mids[check_num])), hip_mids[check_num], label='Hip Mids', color='b')
ax1.set_ylabel('Hip Mids', color='b')
ax2 = ax1.twinx()
ax2.plot(range(len(heel_mids[check_num])), heel_mids[check_num], label='Heel Mids', color='r')
ax2.set_ylabel('Heel Mids', color='r')
ax1.set_title('Hip Mids and Heel Mids Comparison')

plt.subplot(2, 2, 4)
ax1 = plt.gca()
ax1.plot(range(len(hip_ends[check_num])), hip_ends[check_num], label='Hip Ends', color='b')
ax1.set_ylabel('Hip Ends', color='b')
ax2 = ax1.twinx()
ax2.plot(range(len(heel_ends[check_num])), heel_ends[check_num], label='Heel Ends', color='r')
ax2.set_ylabel('Heel Ends', color='r')
ax1.set_title('Hip Ends and Heel Ends Comparison')

plt.show()

