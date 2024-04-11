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
from torch import autograd

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

        sw = SlidingWindow(args.win_len, get_y=0, horizon=args.horizon)
        X_window, y_window = sw(input_data)
        X.extend(X_window)
        y.extend(y_window)

    X = np.array(X)
    y = np.array(y)

    print(f"Input data shape: {input_data.shape}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of targets: {len(y)}")
    fcst_history = 104 # # steps in the past
    fcst_horizon = 60  # # steps in the future
    valid_size   = 0.1  # int or float indicating the size of the training set
    test_size    = 0.2  # int or float indicating the size of the test set

    splits = get_forecasting_splits(pd.Dataframe(X), fcst_history=fcst_history, fcst_horizon=fcst_horizon,
                                    valid_size=valid_size, test_size=test_size)
    x_vars = df.columns[1:]
    y_vars = df.columns[1:]
    X, y = prepare_forecasting_data(pd.DataFrame(X), fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars)
    arch_config = dict(
        n_layers=3,  # number of encoder layers
        n_heads=4,  # number of heads
        d_model=16,  # dimension of model
        d_ff=128,  # dimension of fully connected network
        attn_dropout=0.0, # dropout applied to the attention weights
        dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
        patch_len=24,  # length of the patch applied to the time series to create patches
        stride=2,  # stride used when creating patches
        padding_patch=True,  # padding_patch
    )
    learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models",
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=ShowGraph())
    n_epochs = 100
    lr_max = 0.0025
    learn.fit_one_cycle(n_epochs, lr_max=lr_max)
    learn.export('patchTST.pt')

if __name__ == '__main__':
    main(sys.argv)
