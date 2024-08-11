import depthai 

def main():
    pipeline = depthai.Pipeline()
    # Define sources and outputs
    camRgb = pipeline.create(depthai.node.Camera)
    rgbCamSocket = depthai.CameraBoardSocket.CAM_A
    rgbOut = pipeline.create(depthai.node.XLinkOut)
    rgbOut.setStreamName("rgb")
    camRgb.setBoardSocket(rgbCamSocket)
    camRgb.setSize(1920, 1080) # 1080,720
    camRgb.setFps(10)


    camRgb.setBoardSocket(rgbCamSocket)
    camRgb.video.link(rgbOut.input)

    # Create nodes, configure them and link them together

    # Connect to the device and upload the pipeline to it
    with depthai.Device(pipeline) as device:
        # Print MxID, USB speed, and available cameras on the device
        print('MxId:',device.getDeviceInfo().getMxId())
        print('USB speed:',device.getUsbSpeed())
        print('Connected cameras:',device.getConnectedCameras())
        
        cam_info = device.getOutputQueue("rgb")
        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(rgbCamSocket)
        distortion_coeff = calibData.getDistortionCoefficients(rgbCamSocket)

        print("Camera intrinsics (RGB): ",intrinsics)
        print("Distortion coeff (RGB): ",distortion_coeff)

if __name__ == "__main__":
    main()
