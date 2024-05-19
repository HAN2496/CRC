import numpy as np

# 제공된 period와 amplitude 데이터
periods = np.array([148, 28, 13, 141, 30, 13, 142, 21, 27, 150, 39, 14, 148, 26, 14, 129, 27, 16, 131, 39, 109])
amplitudes = np.array([0.6827763332920007, 43.9317261449531, 5.673119220665498, 2.691987804784201, 44.1568655688967,
                       4.598578554318198, 3.0413721728798, 37.4137918752393, 0.23242375061990117, 11.8171822786008,
                       43.245995368311696, 4.881667247334899, 5.2313878186143015, 45.317098938954004, 1.2496828643518008,
                       3.9391020604040996, 38.2757809188275, 4.8239764982292, 4.797905664268203, 39.5200711166272,
                       7.7277871023303, 26.766654800907098])
sorted_amplitudes = np.sort(amplitudes)
print(sorted_amplitudes)
# 임계값 설정 (예: 전체 period 평균의 50% 이상)
threshold = np.mean(periods) * 0.5

# 임계값보다 큰 period와 해당 amplitude 선택
valid_indices = periods > threshold
filtered_periods = periods[valid_indices]
filtered_amplitudes = amplitudes[valid_indices]

# 평균 주기와 평균 진폭 계산
average_period = np.mean(filtered_periods)
average_amplitude = np.mean(filtered_amplitudes)

# 결과 출력
print("필터링된 주기:", filtered_periods)
print("필터링된 진폭:", filtered_amplitudes)
print("평균 주기:", average_period)
print("평균 진폭:", average_amplitude)
print(len(average_amplitude))