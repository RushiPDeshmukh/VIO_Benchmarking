import depthai as dai
import time 

# Create pipeline
pipeline = dai.Pipeline()

# Create RGB camera node
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)  # Set the frame rate here

# Create XLink output node
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)
prev_time = time.time_ns()
# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()  # Blocking call, will wait until a new data has arrived
        # frame = in_rgb.getCvFrame()
        current_time = time.time_ns()
        print((current_time-prev_time)*1e-9)
        prev_time=current_time
        
        # Display the frame
        # cv2.imshow("rgb", frame)

        # if cv2.waitKey(1) == ord('q'):
        #     break
