#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(500, 400)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)

#integral image
#img = cv2.imread('images\image1.jpg')
# matrix_A = [[1,0,0,1,0],[0,0,0,1,0],[0,1,0,1,0],[1,1,0,1,0],[1,0,0,0,1]]
def integral_image(img):
    img = cv2.imread(img)
    A = np.array(img)
    print(A)
    print(A.shape)
    m,n,n_channels = A.shape
    for i in range(m):
        initial_sum = 0
        for j in range(n):
            initial_sum += A[i][j]
            A[i][j] = initial_sum
            if i > 0:
                A[i][j] += A[i-1][j]
    return A

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    first_frame = 0
    while True:
        videoIn = video.get()
        cv2.imshow("video", videoIn.getCvFrame())
        if cv2.waitKey(1) == ord('c'):
            cv2.imwrite('rgb_image'+str(first_frame)+'.jpg', videoIn.getCvFrame())
            print(integral_image('rgb_image0.jpg'))
            first_frame += 1
        if cv2.waitKey(1) == ord('q'):
            break
        
