# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
# self
import torch_warp


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 先读取图像
frame_1 = cv2.imread("./frame_0016.png")
frame_2 = cv2.imread("./frame_0017.png")
height, width, channel = frame_1.shape

# 读取前后光流
forward_flow  = numpy.load("./forward_flow.npy")
backward_flow = numpy.load("./backward_flow.npy")
print(frame_2.shape, backward_flow.shape)
forward_flow  = forward_flow[:height, :width].copy()
backward_flow = backward_flow[:height, :width].copy()

# 做一次 warp
warp_2to1_using_torch = torch_warp.warp(frame_2, forward_flow)
warp_1to2_using_torch = torch_warp.warp(frame_1, backward_flow)

# 展示
display = numpy.concatenate([warp_2to1_using_torch, warp_1to2_using_torch], axis=0)
cv_show(display)