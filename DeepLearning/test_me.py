import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
total_path = f"datasets/epic/AB06/10_09_18/levelground/gcRight/levelground_ccw_normal_02_01.csv"

def to_polar_coordinates(value):
    theta = (value / 100) * 2 * np.pi
    return (np.cos(theta), np.sin(theta))

a = np.array([1, 2])

print(np.array(to_polar_coordinates(a)))