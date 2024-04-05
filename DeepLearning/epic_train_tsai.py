"""
Estimation 코드
"""
import numpy as np
import os
from tsai.all import *
from fastai.data.all import *
import torch
import random
from utils import to_polar_coordinates, arg_parser, common_arg_parser

#Step0: GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def main(args):
    #parser 받은 값들로 세팅
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    window_length = args.win_len
    stride_num = args.stride_num
    prefix = args.prefix

    #데이터 로드
    X = []
    y = []
    y_scalar = []
    if args.test==False:
        excluded_values = [16, 20, 23]
        random_values = []
        while len(random_values) < 2:
            rand_num = random.randint(0, 24)
            if rand_num not in excluded_values and rand_num not in random_values:
                random_values.append(rand_num)
        prefix = prefix + f'_patient_{random_values[0]}_{random_values[1]}'
    else:
        random_values = [args.pnum1, args.pnum2]
        prefix = prefix + f'_patient_{args.pnum1}_{args.pnum2}'

    for i in random_values:
        path = 'datasets/epic'
        patient_path = f"{path}/AB{str(i+6).zfill(2)}"
        folders_in_patient_path = [item for item in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, item))]
        specific_folder_path = [folder for folder in folders_in_patient_path if folder != "osimxml"][0]

        if i==0:
            gon_path1 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_02_01.csv"
            gon_path2 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_02_01.csv"
            imu_path1 = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_02_01.csv"
            target_path1 = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_ccw_normal_02_01.csv"
            target_path2 = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_cw_normal_02_01.csv"

        else:
            gon_path1 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_01_02.csv"
            gon_path2 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_01_02.csv"
            imu_path1 = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_01_02.csv"
            target_path1 = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_ccw_normal_01_02.csv"
            target_path2 = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_cw_normal_01_02.csv"

        gon1 = pd.read_csv(gon_path1).iloc[::5, 4].values
        imu1 = pd.read_csv(imu_path1).iloc[:, 4].values
        target1 = pd.read_csv(target_path1).iloc[:, 1].values
        target_scalar = target1
        target1 = to_polar_coordinates(target1)

        input2 = pd.read_csv(gon_path2).iloc[::5, 4].values
        target2 = pd.read_csv(target_path2).iloc[:, 1].values
        target2 = to_polar_coordinates(target2)

        print(f"gon shape: {gon1.shape}, imu shape: {imu1.shape}")
        input = np.column_stack((gon1, imu1))
        print(input.shape)
        sw = SlidingWindow(window_length, stride=stride_num, get_y=None)
        X_window, _ = sw(input)
        y_window, _ = sw(target1)
        y_window_scalar, _ = sw(target_scalar)

        X.extend(X_window)
        y.extend(y_window)
        y_scalar.extend(y_window_scalar)


    X = np.array(X)
    y = np.array(y)
    y_scalar = np.array(y_scalar)
    y_scalar = y_scalar.reshape(y_scalar.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    #plt.figure(figsize=(10,10))
    print("y_scalar shape: ", y_scalar.shape)
    plt.plot(y_scalar[:, :])
    plt.show()
    y_scalar = y_scalar.reshape(y.shape[0], -1)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X shape: {X.shape[1]}, y shape: {y.shape[1]}")
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of targets: {len(y)}")

    splits = get_splits(y, valid_size=0.2, stratify=False, random_state=23, shuffle=True, show_plot=False)
    tfms  = [None, [TSRegression()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, num_workers=0)


    # 모델 초기화
    model = InceptionTime(X.shape[1], y.shape[1])

    # Learner 객체 생성
    learn = Learner(dls, model, metrics=rmse)

    if args.test == False:
        # 학습 시작
        learn.fit_one_cycle(800, lr_max=1e-3)
        learn.save(f"{prefix}")
    else:
        #테스트 구간
        learn.load(f"{prefix}")
        learn.model.eval()
        index_num = np.random.randint(len(X))
        example_input_array = X[index_num].reshape(1, X.shape[1], X.shape[2])

        example_input_tensor = torch.tensor(example_input_array, dtype=torch.float)

        # 예측 실행
        with torch.no_grad():
            prediction = learn.model(example_input_tensor)

        print(f"Prediction shape: {prediction.shape}")
        actual_value = y[index_num]

        # 예측 결과와 실제 값 비교
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.plot(prediction[0].numpy(), label='Predicted')
        plt.plot(actual_value, label='Actual')
        plt.title(f'Prediction vs Actual (patient {args.pnum1} and {args.pnum2}) (index num {index_num})')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()

        print(actual_value.shape)

        def xy_to_scaled_value(x, y):
            theta = np.arctan2(y, x)  # 범위: -π to π
            if theta < 0:
                theta += 2 * np.pi  # 범위를 0에서 2π로 조정
            scaled = (theta / (2 * np.pi)) * 100
            return scaled


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
        #plt.plot(y_scalar[index_num], label='Actual')
        plt.plot(actual_value_scalar, label='Actual')
        plt.title(f'Prediction vs Actual in scalar(0 ~ 100) (patient {args.pnum1} and {args.pnum2})')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main(sys.argv)
