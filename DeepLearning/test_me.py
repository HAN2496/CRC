import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
total_path = f"datasets/epic/AB06/10_09_18/levelground/gcRight/levelground_ccw_normal_02_01.csv"

import random

data_num = [i for i in range(25) if i not in [16, 20, 23]]
tmp = []
for i in data_num:
    for j in data_num:
        tmp.append([i, j])
print(tmp)