import torch
from tsai.basics import *

from utils.load_train_data import load_train_data
from utils.td import TD
from src.trajectory_generator import Reference
from configs.config_estimation import MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, INPUT_WINDOW_LENGTH
from utils.visualize import visualize_predictions, visualize_reference_trajectory

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

"""
Step1: Train estimation model
"""
X_train, y_train, X_test, y_test = load_train_data()
splits = TimeSplitter(int(0.2 * len(X_train)), fcst_horizon=0, show_plot=False)(y_train)

# Train model
tfms = [None, TSRegression()]
batch_tfms = TSStandardize()
reg = TSRegressor(X_train, y_train, splits=splits, path='models', tfms=tfms, batch_tfms=batch_tfms, bs=BATCH_SIZE, arch=None, metrics=mae)
reg.to(device)
reg.fit_one_cycle(NUM_EPOCHS, LEARNING_RATE)

# Save model
model_path = f'{MODEL_NAME}_{NUM_EPOCHS}epochs.pth'
reg.export(model_path)
print(f'Model saved to {model_path}')

#Visualize
n_samples = 9
visualize_predictions(reg, X_test, y_test, n_samples=n_samples, input_window_length=INPUT_WINDOW_LENGTH)



"""
Step2:  Fit reference trajectory (by gaussian process)
"""
td = TD(number=6, cut=True)
X = td.datas['heelstrike']
y = td.datas['hip_sagittal']
gp = Reference(fit_mode=True, X=X, y=y)
visualize_reference_trajectory(td.datas, gp)