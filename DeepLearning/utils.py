import numpy as np

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
    parser.add_argument('--stride_num', help="Stride number at Sliding window", type=int, default=100)
    parser.add_argument('--use_imu', help="Use IMU data if True", type=bool, default=True)
    parser.add_argument('--prefix', help="Name of your specific code", type=str, default='test')
    parser.add_argument('--test', help="True if test", type=bool, default=False)
    parser.add_argument('--pnum1', help="Patient number1", type=int, default=0)
    parser.add_argument('--pnum2', help="Patient number2", type=int, default=0)
    return parser