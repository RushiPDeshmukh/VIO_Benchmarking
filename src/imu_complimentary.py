import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('/home/jidnyesha/VIO_Benchmarking/Dataset/imu_dataset/IMU_Data_1.csv')
DELTA_T=0.01 # sec

NUM_READINGS = len(df)//4
df.truncate(after=NUM_READINGS)
# print(NUM_READINGS,df.shape)

# Complementary Filter 
est_roll = np.empty((df.shape[0],1))

A_y_0 = df['Acc_y'][0]/np.sqrt(df['Acc_x'][0]**2 + df['Acc_y'][0]**2 + df['Acc_z'][0]**2)
A_z_0 = df['Acc_z'][0]/np.sqrt(df['Acc_x'][0]**2 + df['Acc_y'][0]**2 + df['Acc_z'][0]**2)    
est_roll[0]= np.arctan2(A_y_0,A_z_0)
A_r = np.empty((df.shape[0],1))

alpha = 0.95 # 0 to 1

for k in tqdm(range(0,NUM_READINGS-1)):
    A_y = df['Acc_y'][k]/np.sqrt(df['Acc_x'][k]**2 + df['Acc_y'][k]**2 + df['Acc_z'][k]**2)
    A_z = df['Acc_z'][k]/np.sqrt(df['Acc_x'][k]**2 + df['Acc_y'][k]**2 + df['Acc_z'][k]**2)
    A_r[k] = np.arctan2(A_y,A_z)
    est_roll[k+1] = alpha*est_roll[k] + (1-alpha)*A_r[k] + alpha*df['Gyro_x'][k]*DELTA_T


plt.figure(figsize=(20,10))
plt.plot(est_roll,label="est")
# plt.plot(np.deg2rad(df['Euler_x']),label="truth")
plt.plot(A_r,label="A_r")
plt.xlabel('Time step [ sec ]')
plt.ylabel('Euler_x = Roll (rad)')
plt.legend()

plt.show()