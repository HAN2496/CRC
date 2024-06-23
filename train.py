import torch
from tsai.basics import *

from utils.load_train_data import load_train_data
from utils.td import TD
from src.trajectory_generator import Reference
from configs.config_estimation import MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, INPUT_WINDOW_LENGTH, ARCH
from utils.visualize import visualize_predictions_scalar, visualize_predictions_polar, visualize_reference_trajectory

#Choose whether train Knee or Hip
is_hip = False
if is_hip:
    control_target_name = "hip"
    control_target = "hip_sagittal"
else:
    control_target_name = "knee"
    control_target = "knee_sagittal"


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

"""
Step1: Train estimation model
"""
X_train, y_train, X_test, y_test = load_train_data(control_target)
splits = TimeSplitter(int(0.2 * len(X_train)), fcst_horizon=0, show_plot=False)(y_train)

# Train model
tfms = [None, TSRegression()]
batch_tfms = TSStandardize()
reg = TSRegressor(X_train, y_train, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=BATCH_SIZE, arch=ARCH, metrics=mae)
reg.to(device)
reg.fit_one_cycle(NUM_EPOCHS, LEARNING_RATE)

# Save model
model_path = f'{MODEL_NAME}_{control_target_name}.pth'
reg.export(model_path)
print(f'Model saved to {model_path}')

#Visualize
n_samples = 9
file_name = f"{MODEL_NAME}_{control_target_name}"
visualize_predictions_scalar(control_target_name, file_name, reg, X_test, y_test, n_samples=n_samples, input_window_length=INPUT_WINDOW_LENGTH)
visualize_predictions_polar(file_name, reg, X_test, y_test, n_samples=n_samples, input_window_length=INPUT_WINDOW_LENGTH)

"""
Step2:  Fit reference trajectory (by gaussian process)
"""
td = TD(number=6, choose_one_dataset=False, extract_walking=False)
td.extract_one_datasets(extract_normal_phase=True)
td.extract_walking_phases(3, 6)
X = td.datas['heelstrike']
y = td.datas[control_target]
filename = f'models/gaussian_process_regressor_{control_target_name}'
gp = Reference(fit_mode=True, X=X, y=y, filename=filename)
visualize_reference_trajectory(control_target, td.datas, gp)
