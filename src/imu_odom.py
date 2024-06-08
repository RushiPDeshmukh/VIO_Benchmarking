import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("calib_imu_motion3.csv")

acc_roll = []
acc_pitch =[]
gyro_roll=[]
gyro_pitch=[]
gyro_yaw=[]
mag_yaw=[]
for i in tqdm(range(len(df))):
    acc_roll.append(np.arctan2(df['Acc_y'][i],np.sqrt(df['Acc_x'][i]**2+df['Acc_z'][i]**2)))
    acc_pitch.append(np.arctan2(df['Acc_z'][i],np.sqrt(df['Acc_x'][i]**2+df['Acc_y'][i]**2)))
    if i==0:
        gyro_roll.append(df['Gyro_x'][0])
        gyro_pitch.append(df['Gyro_y'][0])
        gyro_yaw.append(df['Gyro_z'][0])
        t_prev=df['Gyro_ts'][0]
    else:
        dt = (df['Gyro_ts'][i]-df['Gyro_ts'][i-1])/1000
        gyro_roll=gyro_roll[i-1] + df['Gyro_x']*dt
        gyro_pitch=gyro_pitch[i-1] + df['Gyro_y']*dt
        gyro_yaw=gyro_yaw[i-1] + df['Gyro_z']*dt
    mag_yaw.append(np.arctan2(-df['Mag_x'][i],df['Mag_y'][i]))

# Plot graphs
a_t = np.array(df['Acc_ts'])[:,None]
g_t = np.array(df['Gyro_ts'])[:,None]
m_t = np.array(df['Mag_ts'])[:,None]
acc_roll = np.array(acc_roll)[:,None]
acc_pitch = np.array(acc_pitch)[:,None]
gyro_roll = np.array(gyro_roll)[:,None]
gyro_pitch = np.array(gyro_pitch)[:,None]
gyro_yaw = np.array(gyro_yaw)[:,None]
mag_yaw = np.array(mag_yaw)[:,None]
plt.figure(figsize=(20,20))
# plt.subplot(1,3,1)
# plt.plot(a_t,acc_roll,label="acc_roll")
# plt.plot(g_t,gyro_roll,label="gyro_roll")
# plt.xlabel('Time (ms)')
# plt.ylabel('Roll angle (rads)')
# plt.legend()
# plt.subplot(1,3,2)
# plt.plot(a_t,acc_pitch,label='acc_pitch')
# plt.plot(g_t,gyro_pitch,label='gyro_pitch')
# plt.xlabel('Time (ms)')
# plt.ylabel('Pitch angle (rads)')
# plt.legend()
# plt.subplot(1,3,3)
# plt.plot(g_t,gyro_yaw,label='gyro_yaw')
# plt.plot(m_t,mag_yaw,label='mag_yaw')
# plt.xlabel('Time (ms)')
# plt.ylabel('Yaw angle (rads)')
# plt.legend()

plt.scatter(m_t,np.rad2deg(mag_yaw))
plt.xlabel('Time (ms)')
plt.ylabel('Angle (deg)')
plt.show()