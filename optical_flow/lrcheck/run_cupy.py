# Python
import os
import sys
import time
# 3rd party
import cv2
import cupy
import numpy
import ctypes
from numpy.ctypeslib import ndpointer


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# 先读取图像
frame_1 = cv2.imread("./frame_0016.png")
frame_2 = cv2.imread("./frame_0017.png")
height, width, channel = frame_1.shape

# 读取前后光流
forward_flow  = numpy.load("./forward_flow.npy")
backward_flow = numpy.load("./backward_flow.npy")
forward_flow  = forward_flow[:height, :width].copy()
backward_flow = backward_flow[:height, :width].copy()

# 加载我自己的动态库
lib = ctypes.cdll.LoadLibrary("./crane_warp.so")


def backward_warp(x, flow):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), dtype="uint8")
	lib.backward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c
	)
	return warped

def forward_warp(x, flow):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), dtype="uint8")
	lib.forward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c
	)
	return warped


# 使用 2to1 的光流 backward flow 将第二帧 warp 到第一帧的位置
forward_warp_2to1  = forward_warp(frame_2, backward_flow)
# 使用 1to2 的光流 forward flow, 将第二帧 warp 到第一帧的位置
backward_warp_2to1 = backward_warp(frame_2, forward_flow)
cv_show(numpy.concatenate([forward_warp_2to1, backward_warp_2to1], axis=0))

