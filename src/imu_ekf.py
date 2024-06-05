import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
df = pd.read_csv('/home/jidnyesha/VIO_Benchmarking/Dataset/imu_dataset/IMU_Data_1.csv')
DELTA_T=0.01 # sec

roll_0= df['Euler_x'][0]
pitch_0= df['Euler_y'][0]
yaw_0= df['Euler_z'][0]
print(f'Initial orientation: roll={roll_0}, pitch={pitch_0}, yaw={yaw_0}')

def calculate_euler(Ax,Ay,Az,Mx,My):
    # All inputs need to be np arrays
    roll = np.arctan2(Ay,Az) # about x axis
    pitch = np.arctan2(-Ax,np.sqrt(Ay*Ay+Az*Az)) # about y axis
    yaw = np.arctan2(My,Mx) # about z axis
    return roll,pitch,yaw

all_r,all_p,all_y=calculate_euler(df['Acc_x'],df['Acc_y'],df['Acc_z'],df['Mag_x'],df['Mag_y'])
# print(np.rad2deg(all_r[0]),np.rad2deg(all_p[0]),np.rad2deg(all_y[0]))
# print(df.head())

# Load the data
data = df
# Initial state
x = np.array([1, 0, 0, 0])  # quaternion initial state (w, x, y, z)

# Covariance matrix
P = np.eye(4)

# Process noise
Q = np.eye(4) * 0.01

# Measurement noise
R = np.eye(6) * 0.1

# Identity matrix
I = np.eye(4)

def predict(x, P, gyro, dt):
    # Convert gyro from degrees/s to radians/s
    # gyro = np.deg2rad(gyro)
    
    # Normalize gyro
    omega = np.linalg.norm(gyro)
    if omega > 0:
        dq = Rotation.from_rotvec(gyro * dt).as_quat()
        dq = np.roll(dq, 1)  # to make it (w, x, y, z)
    else:
        dq = np.array([1, 0, 0, 0])

    # Quaternion multiplication
    x = Rotation.from_quat(x) * Rotation.from_quat(dq)
    x = x.as_quat()
    
    # Normalize quaternion
    x /= np.linalg.norm(x)
    
    # Update covariance
    P = P + Q
    
    return x, P

def update(x, P, acc, mag):
    # Normalize accelerometer measurement
    acc = acc #/ np.linalg.norm(acc)
    
    # Normalize magnetometer measurement
    mag = mag #/ np.linalg.norm(mag)
    
    # Construct measurement
    z = np.hstack((acc, mag))
    
    # Compute the predicted measurement from state (x)
    h = np.hstack((Rotation.from_quat(x).apply([0, 0, 1]), Rotation.from_quat(x).apply([1, 0, 0])))
    
    # Measurement Jacobian
    H = np.zeros((6, 4))
    
    # Kalman gain
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    # Update state
    y = z - h
    x = x + K @ y
    
    # Normalize quaternion
    x /= np.linalg.norm(x)
    
    # Update covariance
    P = (I - K @ H) @ P
    
    return x, P

# Time step
dt = 0.01

# Lists to store results
ekf_quats = []

# Run EKF
for i in tqdm(range(len(data))):
    gyro = data[['Gyro_x', 'Gyro_y', 'Gyro_z']].iloc[i].values
    acc = data[['Acc_x', 'Acc_y', 'Acc_z']].iloc[i].values
    mag = data[['Mag_x', 'Mag_y', 'Mag_z']].iloc[i].values
    
    x, P = predict(x, P, gyro, dt)
    x, P = update(x, P, acc, mag)
    
    ekf_quats.append(x)

# Convert to numpy arrays
ekf_quats = np.array(ekf_quats)
dataset_quats = data[['Quat_0', 'Quat_1', 'Quat_2', 'Quat_3']].values

# Plot the results
plt.figure(figsize=(15, 10))
labels = ['w', 'x', 'y', 'z']

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(ekf_quats[:, i], label='EKF')
    plt.plot(dataset_quats[:, i], label='Dataset')
    plt.xlabel('Time step')
    plt.ylabel(f'Quaternion {labels[i]}')
    plt.legend()
    plt.title(f'Quaternion {labels[i]} Comparison')

plt.tight_layout()
plt.show()


