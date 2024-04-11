import numpy as np
import pandas as pd
from fastai.callback.core import Callback

def to_polar_coordinates(value):
    theta = (value / 100) * 2 * np.pi
    xs = np.cos(theta)
    ys = np.sin(theta)
    return np.column_stack((xs, ys))

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--train_num', help="Training Number", type=int, default=10)
    parser.add_argument('--win_len', help="Sliing window Length", type=int, default=100)
    parser.add_argument('--horizon', help="Horizen value", type=int, default=10)
    
    parser.add_argument('--stride_num', help="Stride number at Sliding window", type=int, default=10)
    parser.add_argument('--use_imu', help="Use IMU data if True", type=bool, default=True)
    parser.add_argument('--prefix', help="Name of your model", type=str, default='test')
    parser.add_argument('--test', help="True if test", type=bool, default=False)
    parser.add_argument('--pnum1', help="Patient number1", type=int, default=0)
    parser.add_argument('--pnum2', help="Patient number2", type=int, default=0)
    parser.add_argument('--learn_num', help='Learning number', type=int, default=800)
    parser.add_argument('--arch', help='Architecture of your model', type=str, default="PatchTST")
    return parser

class SaveLearningInfo(Callback):
    def __init__(self, log_dir='training_log.csv'):
        super().__init__()
        self.log_dir = log_dir
        self.log = []

    def after_epoch(self):
        if len(self.learn.recorder.values) > 0:
            record = {
                'epoch': self.epoch,
                'train_loss': self.learn.recorder.losses[-1].item() if len(self.learn.recorder.losses) > 0 else None,
                'valid_loss': self.learn.recorder.values[-1][0] if len(self.learn.recorder.values[-1]) > 0 else None
            }
            self.log.append(record)
            pd.DataFrame(self.log).to_csv(f"{self.log_dir}/log.csv", index=False)
        else:
            print(f"Warning: No recorded values for epoch {self.epoch}.")