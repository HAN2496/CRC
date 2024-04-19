"""
Prediction 코드
"""

import numpy as np
import os
from tsai.all import *
from fastai.data.all import *
import torch
import random
from utils import to_polar_coordinates, arg_parser, common_arg_parser, SaveLearningInfo
from fastai.callback.tensorboard import TensorBoardCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error



def main(args):
    #parser 받은 값들로 세팅
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    window_length = args.win_len
    prefix = args.prefix

    #데이터 로드
    X = []
    y = []
    if args.test==False:
        #Step0: GPU 사용 가능 여부 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

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
            gon_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_02_01.csv"
            imu_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_02_01.csv"
            phase_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_ccw_normal_02_01.csv"
            gon_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_02_01.csv"
            phase_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_cw_normal_02_01.csv"
        else:
            gon_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_01_02.csv"
            imu_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_01_02.csv"
            phase_path_ccw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_ccw_normal_01_02.csv"
            gon_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_01_02.csv"
            imu_path_cw = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_cw_normal_02_01.csv"
            phase_path_cw = f"{patient_path}/{specific_folder_path}/levelground/gcRight/levelground_cw_normal_01_02.csv"

        gon_ccw = pd.read_csv(gon_path_ccw).iloc[::5, 4]
        imu_ccw = pd.read_csv(imu_path_ccw).iloc[:, 4]
        phase_ccw = pd.read_csv(phase_path_ccw).iloc[:, 1]
        input_data = np.concatenate((phase_ccw, gon_ccw.values))
        print(f"input data shape: {input_data.shape}")

        sw = SlidingWindow(window_length, stride= args.stride_num, get_y=[0], horizon=args.horizon)
        X_window, y_window = sw(input_data)
        print(f"X_window shape: {X_window.shape} / Y_window shape: {y_window.shape}")
        X.extend(X_window)
        y.extend(y_window)

    X = np.array(X)
    y = np.array(y)
    splits = TimeSplitter(valid_size=0.2)(y)
    tfms = [None, TSForecasting()]

    batch_tfms = TSStandardize()
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")

    learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models", arch="PatchTST", metrics=[mse, mae])
    print(learn.summary())
    if args.test == False:
        # 학습 시작
        os.makedirs('models/prediction/', exist_ok=True)
        learn.fit_one_cycle(800, 1e-3)
        learn.export(f"prediction/{prefix}.pt")
        print(f"Finish Learning. Model is at prediction/{prefix}")
    else:
        #테스트 결과 플롯
        learn = load_learner(f"prediction/{prefix}.pt")
        scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
        scaled_preds = to_np(scaled_preds)
        print(f"scaled_preds.shape: {scaled_preds.shape}")

        scaled_y_true = y[splits[1]]
        results_df = pd.DataFrame(columns=["mse", "mae"])
        results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
        results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
        results_df

        preds, targets = learn.get_preds(dl=dls.valid)
        
        a = 2
        b = 3

        fig, axs = plt.subplots(a, b, figsize=(b*6, a*4))

        for i in range(a):
            for j in range(b):
                index = i * b + j
                if index < len(preds):
                    axs[i, j].plot(preds[index], label="Prediction")
                    axs[i, j].plot(targets[index], label="Real", alpha=0.5)
                    axs[i, j].set_title(f"Sample {index+1}")
                    axs[i, j].legend()
                else: 
                    axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    main(sys.argv)
