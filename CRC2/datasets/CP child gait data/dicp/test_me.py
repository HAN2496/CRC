import matplotlib.pyplot as plt
import pandas as pd
import c3d

filename = "DiCP3a.c3d"

datasets = c3d.Reader(open(filename, 'rb'))
print(datasets)