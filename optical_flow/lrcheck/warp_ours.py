# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
# self
import torch_warp
import ctypes
from numpy.ctypeslib import ndpointer


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

# 加载我自己的动态库
lib = ctypes.cdll.LoadLibrary("./crane_warp.so")
def crane_warp(x, flow):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), dtype="uint8")
	lib.fast_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c
	)
	return warped

# 做一次 warp
warp_2to1_using_crane = crane_warp(frame_2, forward_flow)
warp_1to2_using_crane = crane_warp(frame_1, backward_flow)
display = numpy.concatenate([warp_2to1_using_crane, warp_1to2_using_crane], axis=0)
cv_show(display)

# 计算遮挡
# 如果是原始的计算遮挡的方式, 就是做两次 warp
warp_2to1to2_using_crane = crane_warp(warp_2to1_using_crane, backward_flow)
warp_1to2to1_using_crane = crane_warp(warp_1to2_using_crane, forward_flow)
# 分别做两次减法, 设定阈值
mask_threshold = 25
forward_occulusion = numpy.abs(warp_1to2to1_using_crane.astype("float32") - frame_1.astype("float32"))
forward_occulusion = numpy.mean(forward_occulusion, axis=-1)
forward_occulusion[forward_occulusion < mask_threshold] = 0
forward_occulusion[forward_occulusion > mask_threshold] = 255
forward_occulusion = forward_occulusion.astype("uint8")
backward_occulusion = numpy.abs(warp_2to1to2_using_crane.astype("float32") - frame_2.astype("float32"))
backward_occulusion = numpy.mean(backward_occulusion, axis=-1)
backward_occulusion[backward_occulusion < mask_threshold] = 0
backward_occulusion[backward_occulusion > mask_threshold] = 255
backward_occulusion = backward_occulusion.astype("uint8")

display = numpy.concatenate([forward_occulusion, backward_occulusion], axis=0)
cv_show(display)


# 采用快速计算遮挡方法
def crane_compute_occulusion(fore, back, dis_threshold=1.0):
	h, w, d = fore.shape
	occulusion = numpy.zeros((h, w), dtype="uint8")
	lib.fast_compute_occulusion(
		occulusion.ctypes.data_as(ctypes.c_char_p), 
		fore.ctypes.data_as(ctypes.c_char_p), 
		back.ctypes.data_as(ctypes.c_char_p), 
		h, w, d,
		# 浮点数最好显示转换
		ctypes.c_float(dis_threshold)
	)
	return occulusion

forward_occulusion  = crane_compute_occulusion(forward_flow,  backward_flow)
backward_occulusion = crane_compute_occulusion(backward_flow, forward_flow)

# 展示
display = numpy.concatenate([forward_occulusion, backward_occulusion], axis=0)
cv_show(display)

