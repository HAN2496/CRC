import torch
from tsai.basics import *

from utils.td import TD
from utils.cp import CP
from utils.datasets import Datasets
from utils.load_train_data import load_train_data

from src.trajectory_generator import Reference
from src.controller import Controller
from configs.config_estimation import MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, INPUT_WINDOW_LENGTH


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

"""
Step1: Load estimation model
"""
X_train, y_train, X_test, y_test = load_train_data()

splits = TimeSplitter(int(0.2 * len(X_train)), fcst_horizon=0, show_plot=False)(y_train)

tfms = [None, TSRegression()]
batch_tfms = TSStandardize()
model = TSRegressor(X_train, y_train, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=BATCH_SIZE, arch=None, metrics=mae)
model.to(device)
# _, _, pred = model.get_X_preds(data)

"""
Step2:  Load reference trajectory (by gaussian process)
"""
td = TD(number=6, extract_walking=True)
reference = Reference()

"""
Step3: Control
"""
original_datas = td.datas
total_data_num = len(original_datas)
corrected_datas = Datasets()
controller = Controller(subject=td)

idx = 0
collect_one_more = True
while idx < total_data_num:
    if idx <= INPUT_WINDOW_LENGTH:
        idx += 1
        continue
    elif original_datas['section'][idx] == 0:
        corrected_datas.appends(original_datas.indexs(idx))
    else:
        if collect_one_more:
            corrected_datas.appends(original_datas.indexs(idx))
            reference.update(original_datas['header'][:idx], original_datas['heelstrike'][:idx])
            collect_one_more = False
            control_start_idx = idx
            idx+= 1
            print("Now control start")
            continue
        print(f"Now idx: {idx} (total: {total_data_num})")
        desired_kinematics = controller.control(corrected_datas, original_datas[idx:idx+controller.control_interval])

        for idx1, (x_t, v_t, a_t) in enumerate(desired_kinematics):            
            window = corrected_datas['heelstrike'][-1-INPUT_WINDOW_LENGTH:-1].reshape(1, INPUT_WINDOW_LENGTH, -1)
            _, _, estimated_heelstrike = model.get_X_preds(window)
            radians = (estimated_heelstrike / 100.0) * 2 * np.pi
            estimated_heelstrike_x = np.cos(radians)
            estimated_heelstrike_y = np.sin(radians)

            data_point = {
                'section': original_datas['section'][idx],
                'header': original_datas['header'][idx],
                'hip_sagittal': x_t,
                'hip_sagittal_speed': v_t,
                'hip_sagittal_acc': a_t,
                'heelstrike': original_datas['heelstrike'][idx],
                'heelstrike_x': original_datas['heelstrike_x'][idx],
                'heelstrike_y': original_datas['heelstrike_y'][idx],
                'torque': original_datas[idx + idx1]
            }
            corrected_datas.appends(data_point)
            idx += 1
