import torch
from tsai.basics import *

from utils.td import TD
from utils.cp import CP
from utils.datasets import Datasets
from utils.load_train_data import load_train_data
from utils.visualize import visulize_system
from utils.utils import rad_to_scalar

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


"""
Step2:  Load reference trajectory (by gaussian process)
"""
td = TD(number=6, choose_one_dataset=False, extract_walking=False)
td.extract_one_datasets(True)
td.extract_walking_phases(section_start=4, interval=6)
reference = Reference()

"""
Step3: Control
"""

original_datas = td.datas
total_data_num = len(original_datas)
corrected_datas = Datasets()
controller = Controller(subject=td, reference=reference)

idx = 0
tmp = 0
collect_one_more = True
print("="*50)
while idx < total_data_num:
    if idx <= INPUT_WINDOW_LENGTH:
        corrected_datas.appends(original_datas.indexs(idx))
        idx += 1
    elif original_datas['section'][idx] <= 2:
        corrected_datas.appends(original_datas.indexs(idx))
        idx += 1
    else:
        if collect_one_more:
            corrected_datas.appends(original_datas.indexs(idx))
            section_now = corrected_datas['section'][-1]
            section_now_datas = original_datas.sections(section_now)
            last_section_num = len(original_datas.sections(section_now, index=True))
            reference.update(section_now_datas['header'][:-1], section_now_datas['heelstrike'][:-1])
            plt.plot(original_datas['header'], original_datas['hip_sagittal'])
            plt.plot(reference.times, reference.y_pred, linestyle='--')
            plt.show()
            collect_one_more = False
            control_start_idx = idx
            idx+= 1
            print("Now control start")
            continue
        if tmp < 0:
            corrected_datas.appends(original_datas.indexs(idx))
            idx+= 1
            tmp+=1
        print(f"Now Time: {original_datas['header'][idx]}, idx: ({idx}/{total_data_num})")
        desired_kinematics, scale_x, scale_y = controller.control(corrected_datas, original_datas['torque'][idx:idx+controller.control_interval])

        for idx1, (x_t, v_t, a_t) in enumerate(desired_kinematics):
            # window = np.array(corrected_datas['heelstrike'][-1-INPUT_WINDOW_LENGTH:-1]).reshape(1, INPUT_WINDOW_LENGTH, -1)
            # _, _, estimated_heelstrike = model.get_X_preds(window)
            # estimated_heelstrike = estimated_heelstrike[0]
            # estimated_heelstrike_x, estimated_heelstrike_y, estimated_heelstrike = rad_to_scalar(estimated_heelstrike[0], estimated_heelstrike[1])
            # estimated_heelstrike = np.clip(estimated_heelstrike, 0, 100)
            # radians = (estimated_heelstrike / 100.0) * 2 * np.pi
            # estimated_heelstrike_x = np.cos(radians)
            # estimated_heelstrike_y = np.sin(radians)
            estimated_heelstrike = original_datas['heelstrike'][idx]
            estimated_heelstrike_x = original_datas['heelstrike_x'][idx]
            estimated_heelstrike_y = original_datas['heelstrike_y'][idx]

            data_point = {
                'section': original_datas['section'][idx],
                'header': original_datas['header'][idx],
                'hip_sagittal': x_t,
                'hip_sagittal_v': v_t,
                'hip_sagittal_a': a_t,
                'heelstrike': estimated_heelstrike,
                'heelstrike_x': estimated_heelstrike_x,
                'heelstrike_y': estimated_heelstrike_y,
                'torque': original_datas[idx + idx1]
            }
            if idx == total_data_num - 1:
                visulize_system(idx, original_datas, corrected_datas, reference, (scale_x, scale_y))
            else:
                visulize_system(idx, original_datas, corrected_datas, reference, (scale_x, scale_y), last=True)
            corrected_datas.appends(data_point)
            idx += 1
