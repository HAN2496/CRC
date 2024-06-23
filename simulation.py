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

#Choose whether control Knee or Hip
is_hip = False
if is_hip:
    control_target_name = ""
    control_target = "hip_sagittal"
    control_target_v = "hip_sagittal_v"
    control_target_a = "hip_sagittal_a"
    control_target_t = "hip_sagittal_torque"
    num = 4
else:
    control_target_name = "_knee"
    control_target = "knee_sagittal"
    control_target_v = "knee_sagittal_v"
    control_target_a = "knee_sagittal_a"
    control_target_t = "knee_sagittal_torque"
    num = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

"""
Step1: Load estimation model
"""
model_path = f'models/{MODEL_NAME}_knee.pth'
print(f"model path: {model_path}")
model = load_learner(model_path)

"""
Step2:  Load reference trajectory (by gaussian process)
"""
td = TD(number=6, choose_one_dataset=False, extract_walking=False)
td.extract_one_datasets(True, num=num)
td.extract_walking_phases(section_start=3, interval=4)
reference = Reference(filename=f'models/gaussian_process_regressor{control_target_name}')

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
    elif original_datas['section'][idx] <= 1:
        corrected_datas.appends(original_datas.indexs(idx))
        idx += 1
    else:
        if collect_one_more:
            corrected_datas.appends(original_datas.indexs(idx))
            section_now = corrected_datas['section'][-1]
            section_now_datas = original_datas.sections(section_now)
            last_section_num = len(original_datas.sections(section_now, index=True))
            reference.update(section_now_datas['header'][:-1], section_now_datas['heelstrike'][:-1])
            plt.plot(original_datas['header'], original_datas[control_target])
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
            continue
        print(f"Now Time: {original_datas['header'][idx]}, idx: ({idx}/{total_data_num})")
        desired_kinematics, scale_x, scale_y = controller.control(control_target, corrected_datas, original_datas[control_target_t][idx-1:idx+controller.control_interval-1])

        for idx1, (x_t, v_t, a_t) in enumerate(desired_kinematics):
            window = np.array(corrected_datas[control_target][-1-INPUT_WINDOW_LENGTH:-1]).reshape(1, INPUT_WINDOW_LENGTH, -1)
            _, _, estimated_heelstrikes = model.get_X_preds(window)
            estimated_heelstrike_x, estimated_heelstrike_y = estimated_heelstrikes[0]
            estimated_heelstrike = rad_to_scalar(estimated_heelstrike_x, estimated_heelstrike_y)

            print(f"Estimated heelstrike: {round(estimated_heelstrike, 2)} / True heelstrike: {round(original_datas['heelstrike'][idx], 2)}")
            # estimated_heelstrike = original_datas['heelstrike'][idx]
            # estimated_heelstrike_x = original_datas['heelstrike_x'][idx]
            # estimated_heelstrike_y = original_datas['heelstrike_y'][idx]

            data_point = {
                'section': original_datas['section'][idx],
                'header': original_datas['header'][idx],
                control_target: x_t,
                control_target_v: v_t,
                control_target_a: a_t,
                'heelstrike': estimated_heelstrike,
                'heelstrike_x': estimated_heelstrike_x,
                'heelstrike_y': estimated_heelstrike_y,
                control_target_t: original_datas[idx + idx1]
            }
            if idx == total_data_num - 1:
                visulize_system(idx, control_target, original_datas, corrected_datas, reference, (scale_x, scale_y), name='control', last=True)
            else:
                visulize_system(idx, control_target, original_datas, corrected_datas, reference, (scale_x, scale_y))
            corrected_datas.appends(data_point)
            idx += 1
