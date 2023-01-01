# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
import torch
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

# 数据都改成 torch, 放到 GPU 上
frame_1 = torch.as_tensor(frame_1).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
frame_2 = torch.as_tensor(frame_2).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
forward_flow  = torch.as_tensor(forward_flow).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
backward_flow = torch.as_tensor(backward_flow).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()


# 提前选一个结果
warp_2to1_using_crane = torch.zeros(frame_1.shape, dtype=torch.uint8).cuda()
warp_1to2_using_crane = torch.zeros(frame_2.shape, dtype=torch.uint8).cuda()
import fast_optical_flow
fast_optical_flow.warp(warp_2to1_using_crane, frame_2, forward_flow)
fast_optical_flow.warp(warp_1to2_using_crane, frame_1, backward_flow)
# 做前后 warp
warp_2to1_using_crane = warp_2to1_using_crane.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
warp_1to2_using_crane = warp_1to2_using_crane.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
cv_show(numpy.concatenate([warp_2to1_using_crane, warp_1to2_using_crane], axis=0))

# 快速计算遮挡
dis_threshold = 2.0
forward_occulusion = torch.zeros((1, height, width), dtype=torch.uint8).cuda()
fast_optical_flow.lrcheck(forward_occulusion, forward_flow, backward_flow, dis_threshold)
forward_occulusion = forward_occulusion.detach().cpu().squeeze(0).numpy()

backward_occulusion = torch.zeros((1, height, width), dtype=torch.uint8).cuda()
fast_optical_flow.lrcheck(backward_occulusion, backward_flow, forward_flow, dis_threshold)
backward_occulusion = backward_occulusion.detach().cpu().squeeze(0).numpy()
cv_show(numpy.concatenate([forward_occulusion, backward_occulusion], axis=0))
