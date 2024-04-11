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

def main(args):
    #parser 받은 값들로 세팅
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    window_length = args.win_len
    stride_num = args.stride_num
    prefix = args.prefix
    shift = -5

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
            gon_path1 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_02_01.csv"
            gon_path2 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_02_01.csv"
            imu_path1 = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_02_01.csv"

        else:
            gon_path1 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_ccw_normal_01_02.csv"
            gon_path2 = f"{patient_path}/{specific_folder_path}/levelground/gon/levelground_cw_normal_01_02.csv"
            imu_path1 = f"{patient_path}/{specific_folder_path}/levelground/imu/levelground_ccw_normal_01_02.csv"

        gon1 = pd.read_csv(gon_path1).iloc[::5, 4]
        imu1 = pd.read_csv(imu_path1).iloc[:, 4]
        input_data = np.column_stack((gon1.values, imu1.values))

        sw = SlidingWindow(window_length, stride=stride_num, get_y=0, horizon=args.horizon)
        X_window, y_window = sw(input_data)
        X.extend(X_window)
        y.extend(y_window)

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of targets: {len(y)}")

    splits = get_splits(np.arange(len(X)), valid_size=0.2, stratify=False, random_state=23, shuffle=True, show_plot=False)
    dsets = TSDatasets(X, y, tfms=[None, [TSForecasting()]], splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, num_workers=0)
    
    # 모델 초기화
    model = PatchTST(c_in=X.shape[1], c_out=1, seq_len=args.win_len)

    # Learner 객체 생성
    callbacks = SaveLearningInfo(log_dir='models/prediction')
    learn = Learner(dls, model, metrics=rmse, cbs=[callbacks])

    if args.test == False:
        # 학습 시작
        os.makedirs('models/prediction/', exist_ok=True)
        learn.fit_one_cycle(args.learn_num, lr_max=1e-3)
        learn.save(f"prediction/{prefix}")
        print(f"Finish Learning. Model is at prediction/{prefix}")
    else:
        #테스트 결과 플롯
        learn.load(f"prediction/{prefix}")
        learn.model.eval()

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
