import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

filename = "EPIC/3a_RHipAngles_z.txt"
file = pd.read_csv(filename).values

dataset = []
for data in file:
    dataset.append(data[0])
print(dataset)

# plt.plot(range(len(dataset)), dataset)
# plt.show()

df = pd.DataFrame(dataset)
df.to_csv('EPIC/3a_RHipAngles_z.csv', index=False)

gon_path = "EPIC/AB06/gon/levelground_ccw_normal_02_01.csv"
heelstrike_path = "EPIC/AB06/gcRight/levelground_ccw_normal_02_01.csv"

df_gon = pd.read_csv(gon_path)['hip_sagittal'][::5].values
df_heelstrike = pd.read_csv(heelstrike_path)['HeelStrike'].values

fig, ax1 = plt.subplots()

ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Hip Sagittal Angle', color='tab:blue')
ax1.plot(range(len(df_gon)), df_gon, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Heel Strike', color='tab:red')
ax2.plot(range(len(df_heelstrike)), df_heelstrike, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.show()