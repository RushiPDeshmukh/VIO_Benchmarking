import depthai as dai
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.figure(figsize=(20,20))
ax = plt.subplot(111,projection='polar')
ax.set_yticklabels([])
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
line, =ax.plot([],[],'ro')

def update_plot(yaw_data):
    # yaw in radians
    # yaw_deg = np.rad2deg(yaw_data)
    line.set_data(yaw_data,np.ones_like(yaw_data))
    plt.draw()
    plt.pause(0.001) # 1 ms 

def calculate_yaw(mx,my,mz):
    return np.arctan2(-mz,my)

def calculate_roll_pitch(ax,ay,az):
    roll = np.arctan2(ay,np.sqrt(ax**2+az**2))
    pitch = np.arctan2(az,np.sqrt(ax**2+ay**2))
    return roll,pitch

def complementary_filter(ax,ay,az,gx,gy,dt,prev_roll=None,prev_pitch=None):
    acc_roll = np.arctan2(ay,np.sqrt(ax**2+az**2))
    acc_pitch= np.arctan2(az,np.sqrt(ax**2+ay**2))
    if dt==0:
        print("NO DT! ")
        return 0,0
    if prev_roll is None:
        gyro_roll = gx
    else:
        gyro_roll=prev_roll+gx*(dt/1000)
    if prev_pitch is None:
        gyro_pitch = gy
    else:
        gyro_pitch=prev_pitch+gy*(dt/1000)

    alpha=1
    comp_roll = alpha*gyro_roll+(1-alpha)*acc_roll
    comp_pitch = alpha*gyro_pitch + (1-alpha)*acc_pitch    
    return comp_roll,comp_pitch


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

# enable ACCELEROMETER_RAW at 100 hz rate
# imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 400)
# # enable GYROSCOPE_RAW at 100 hz rate
# imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
# # enable MAGNETOMETER_RAW at 100 hz rate
# imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_RAW, 100)

# Enable calibrated(check) sensors
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER,500)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED,400)
imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_CALIBRATED,100)

# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

# Link plugins IMU -> XLINK
imu.out.link(xlinkOut.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:

    def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds()*1000

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    baseTs = None
    plt.ion()
    prev_roll=None
    prev_pitch=None
    prev_gyro_ts = 0
    while True:
        imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived

        imuPackets = imuData.packets
        # print(dir(imuPackets[0]))
        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter
            gyroValues = imuPacket.gyroscope
            magValues = imuPacket.magneticField
            acceleroTs = acceleroValues.getTimestampDevice()
            gyroTs = gyroValues.getTimestampDevice()
            magTs=magValues.getTimestampDevice()
            if baseTs is None:
                baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
            acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
            gyroTs = timeDeltaToMilliS(gyroTs - baseTs)
            magTs = timeDeltaToMilliS(magTs-baseTs)

            yaw = calculate_yaw(magValues.x,magValues.y,magValues.z)
            roll,pitch = calculate_roll_pitch(acceleroValues.x,acceleroValues.y,acceleroValues.z)
            dt = gyroTs-prev_gyro_ts
            comp_roll,comp_pitch = complementary_filter(acceleroValues.x,acceleroValues.y,acceleroValues.z,gyroValues.x,gyroValues.y,dt,prev_roll,prev_pitch)
            prev_roll = comp_roll
            prev_pitch = comp_pitch
            prev_gyro_ts = gyroTs
            # print(f"YAW X: {magValues.x}")
            # print(f"YAW Y: {magValues.y}")
            update_plot([comp_roll])

    plt.ioff()
    plt.show()

