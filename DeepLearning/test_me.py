import numpy as np
import pandas as pd
from tsai.all import *
from sklearn.model_selection import train_test_split
wl = 6
stride = 5

t = np.repeat(np.arange(13).reshape(-1,1), 3, axis=-1)
print('input shape:', t.shape)
X, y = SlidingWindow(wl, stride=stride, pad_remainder=True, get_y=[])(t)
print(X.shape)