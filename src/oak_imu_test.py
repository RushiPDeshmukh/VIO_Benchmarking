#!/usr/bin/env python3
import cv2
import depthai as dai
import pandas as pd
import math
# from matplotlib.pyplot import 
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

# enable ACCELEROMETER_RAW at 100 hz rate
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
# enable GYROSCOPE_RAW at 100 hz rate
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
# enable MAGNETOMETER_RAW at 100 hz rate
imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_RAW, 100)

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
            imuF = "{:.06f}"
            tsF  = "{:.03f}"
            print(f"Base Ts = {(baseTs)} ms")
            print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
            print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")
            print(f"Magnetometer timestamp: {tsF.format(magTs)} ms")
            print(f"Magnetometer [uT]: x: {imuF.format(magValues.x)} y: {imuF.format(magValues.y)} z: {imuF.format(magValues.z)} ")
            print(f"Delta : Acc & Gyro = {tsF.format(acceleroTs -gyroTs)} ms , Gyro to Mag {tsF.format(gyroTs- magTs)} ms")
            print("------------------------------------------------------------------------------------------------------------------------")

        if cv2.waitKey(1) == ord('q'):
            break
