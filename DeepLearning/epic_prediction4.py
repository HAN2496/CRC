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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split


def plot_validation_data(learner):
    val_dl = learner.dls.valid_ds
    preds = []
    actuals = []

    for xb, yb in val_dl:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            pred = learner.model(xb)
            preds.append(pred.cpu().numpy())
            actuals.append(yb.cpu().numpy())

    predictions = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:100], label='Actual Validation Data')
    plt.plot(predictions[:100], label='Predicted Validation Data')
    plt.title('Validation Data: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Data Value')
    plt.legend()
    plt.show()

def check_data_loader(dl):
    try:
        xb, yb = next(iter(dl))
        print(f"Data batch shape: {xb.shape}, Labels batch shape: {yb.shape}")
    except StopIteration:
        print("No data available in the DataLoader.")
    except Exception as e:
        print(f"An error occurred: {e}")



#Step0: GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def main(args):
    #parser 받은 값들로 세팅
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    window_length = args.win_len

    #데이터 로드
    X = []
    y = []
    if args.test==False:


        excluded_values = [16, 20, 23]
        random_values = []
        while len(random_values) < 2:
            rand_num = random.randint(0, 24)
            if rand_num not in excluded_values and rand_num not in random_values:
                random_values.append(rand_num)
        prefix = args.prefix + f'_patient_{random_values[0]}_{random_values[1]}'
    else:

        random_values = [args.pnum1, args.pnum2]
        prefix = args.prefix + f'_patient_{args.pnum1}_{args.pnum2}'

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
        input_data = np.concatenate((phase_ccw.values, gon_ccw.values))
        print(f"input data shape: {input_data.shape}")

        sw = SlidingWindow(window_length, stride= args.stride_num, get_y=[0], horizon=args.horizon) #Stride: 건너뛰는 양 / horizon: 예측하고 싶은 미래 step수
        X_window, y_window = sw(input_data)
        print(f"X_window shape: {X_window.shape} / Y_window shape: {y_window.shape}")
        X.extend(X_window)
        y.extend(y_window)

    X = np.array(X)
    y = np.array(y)
    print(f"X shape: {X.shape} / Y shape: {y.shape}")


    #splits = TimeSplitter(show_plot=False)(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    splits = (list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_valid))))

    #splits = get_splits(y, valid_size=0.2, stratify=False, random_state=23, shuffle=True, show_plot=False)
    tfms  = [None, [TSForecasting()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    batch_size = 64  # Adjust the batch size if needed
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=batch_size, num_workers=0)
    dls.show_batch(figsize=(10, 6))
    print(f'Train set size: {len(dls.train_ds)}')
    print(f'Valid set size: {len(dls.valid_ds)}')
    print(f'Valid set size: {len(dls.valid)}')

    model = InceptionTimePlus(X.shape[1], y.shape[1])
    #model = PatchTST(X.shape[1], y.shape[1], seq_len = args.horizon)
    learn = Learner(dls, model, metrics=rmse)
    #learn = TSForecaster(X, y, splits=splits, batch_size=16, path="models", arch="PatchTST", metrics=[mse, mae])
    get_arch(learn)
    if args.test == False:
        os.makedirs('models/prediction/', exist_ok=True)
        learn.fit_one_cycle(args.learn_num, 1e-3)
        learn.export(f"models/prediction/{prefix}")

        learn.export(f"models/prediction/{prefix}.pt")
        learn.export(f"models/prediction/{prefix}.pth")
        learn.export(f"models/prediction/{prefix}.pkl")
        print(f"Finish Learning. Model is at prediction/{prefix}")
    else:
        #테스트 구간
        itp = load_learner(f"models/prediction/{prefix}.pkl")
        # DataLoader 검사 실행
        check_data_loader(dls.valid_ds)
        plot_validation_data(learn)
        
        probas, _, preds = learn.get_X_preds(X[splits[1]])
        print(skm.mean_squared_error(y[splits[1]], preds, squared=False))

        actuals = []
        preds = []

        # Retrieve and process each batch from the validation DataLoader
        print(learn.dls)
        for xb, yb in learn.dls.valid:
            print(xb, yb)
            with torch.no_grad():
                pred = learn.model(xb)  # Generate predictions
            preds.extend(pred.cpu().numpy())  # Store predictions
            actuals.extend(yb.cpu().numpy())  # Store actual values

        # Convert lists to arrays for plotting
        actuals = np.array(actuals).flatten()
        preds = np.array(preds).flatten()

        # Plotting predictions vs actuals
        print(actuals.shape, preds.shape)
        plot_predictions(preds, actuals, title="Test Predictions vs Actuals", save_path=f"{prefix}_prediction_comparison.png")
  
if __name__ == '__main__':
    main(sys.argv)
