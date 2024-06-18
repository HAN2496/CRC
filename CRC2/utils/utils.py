import numpy as np

def rad_to_scalar(x, y):
    angle_radians = np.arctan2(y, x)
    if angle_radians < 0:
        angle_radians += 2 * np.pi
    scalar = (angle_radians / (2 * np.pi)) * 100
    return x, y, scalar
