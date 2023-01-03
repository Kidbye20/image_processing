# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
import ctypes
from numpy.ctypeslib import ndpointer


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)
add_to_save = lambda x: os.path.join(save_dir, x)


# 先读取图像
frame_1 = cv2.imread("./frame_0016.png")
frame_2 = cv2.imread("./frame_0017.png")
height, width, channel = frame_1.shape

# 读取前后光流
forward_flow  = numpy.load("./forward_flow.npy")
backward_flow = numpy.load("./backward_flow.npy")
forward_flow  = forward_flow[:height, :width].copy()
backward_flow = backward_flow[:height, :width].copy()

# 可视化光流
# https://github.com/xiaofeng94/GMFlowNet/blob/master/core/utils/flow_viz.py
import flow_viz
forward_flow_visualize  = flow_viz.flow_to_image(forward_flow)[:, :, ::-1] # [:, :, ::-1] 是为了 opencv 显示 BGR 序
backward_flow_visualize = flow_viz.flow_to_image(backward_flow)[:, :, ::-1]
cv_write(add_to_save("forward_flow_visualize.png"),  forward_flow_visualize)
cv_write(add_to_save("backward_flow_visualize.png"), backward_flow_visualize)
cv_show(numpy.concatenate([forward_flow_visualize, backward_flow_visualize], axis=0))


# 编译 C++ 代码用于 forward warp
warp_lib_path = "./crane_warp.so"
os.system("g++ -fPIC -shared -O2 ./crane_warp.cpp -o {}".format(warp_lib_path))

# 加载动态库
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)


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
cv_write(add_to_save("forward_warp_2to1.png"),  forward_warp_2to1)
cv_write(add_to_save("backward_warp_2to1.png"), backward_warp_2to1)
cv_show(numpy.concatenate([forward_warp_2to1, backward_warp_2to1], axis=0))



# 使用 1to2 的光流 forward flow 将第一帧 warp 到第二帧的位置
forward_warp_1to2  = forward_warp(frame_1, forward_flow)
backward_warp_1to2 = backward_warp(frame_1, backward_flow)
cv_write(add_to_save("forward_warp_1to2.png"),  forward_warp_1to2)
cv_write(add_to_save("backward_warp_1to2.png"), backward_warp_1to2)
cv_show(numpy.concatenate([forward_warp_1to2, backward_warp_1to2], axis=0))


# 利用前向光流和反向光流(默认用 backward warp), 计算光流一致性的的差异来计算遮挡
def compute_occlusion_using_value(frame, frame_cycle, mask_threshold=25):
	# 数据要转换一下, 因为 uint8 不支持负数
	occulusion = numpy.abs(frame_cycle.astype("int16") - frame.astype("int16"))
	occulusion = numpy.mean(occulusion, axis=-1)
	occulusion[occulusion < mask_threshold] = 0
	occulusion[occulusion > mask_threshold] = 255
	return occulusion.astype("uint8")


occulusion_value_threshold = 25
# 计算 1to2to1 的遮挡, 得到的是第 1 帧有但第 2 帧没有的内容(注意使用的是 forward_flow, 插值到第 1 帧的场景, 需要去第 2 帧中定位)
backward_warp_1to2to1           = backward_warp(backward_warp_1to2, forward_flow)
forward_occulusion_using_value  = compute_occlusion_using_value(frame_1, backward_warp_1to2to1, mask_threshold=occulusion_value_threshold)
# 计算 2to1to2 的遮挡, 得到的是第 2 帧有但第 1 帧没有的内容
backward_warp_2to1to2           = backward_warp(backward_warp_2to1, backward_flow)
backward_occulusion_using_value = compute_occlusion_using_value(frame_2, backward_warp_2to1to2, mask_threshold=occulusion_value_threshold)
cv_write(add_to_save("forward_occulusion_using_value.png"),  forward_occulusion_using_value)
cv_write(add_to_save("backward_occulusion_using_value.png"), backward_occulusion_using_value)
cv_show(numpy.concatenate([forward_occulusion_using_value, backward_occulusion_using_value], axis=0))
######################################  分析  ############################################
'''
	利用前后光流 warp 的值来判断有一些缺点:
		1. 图像经过了两次 warp(插值), 高频丢失很严重, 这时候再通过值的差异来计算遮挡, 容易检测出原本不是遮挡的地方
		2. 前景(左右视察更大, 离摄像头更近的物体), 如果和背景的颜色很接近, 即使光流算对了, 通过值的差异来计算遮挡,
		   会把被遮挡的部分看成是背景(cycle 回去产生的伪影), 伪影和背景色很接近, 导致检测不出原本是遮挡的地方
		3. 速度慢, 每次 warp 要遍历 H * W 次, 每个点插值一次, 求三个值(RGB)
		4. 对于同色区域、平坦区域(或者说处在同一平面)的光流, 即使光流算错, 两次 warp 之后的值在这一块都是接近的, 原本光流的错误也无法检测出来
'''


# 利用前向光流和反向光流(默认用 backward warp), 计算光流一致性的位置变化来计算遮挡
def compute_occulusion_using_pos(fore, back, mask_threshold=1.0):
	h, w, d = fore.shape
	occulusion = numpy.zeros((h, w), dtype="uint8")
	lib.fast_compute_occulusion(
		occulusion.ctypes.data_as(ctypes.c_char_p), 
		fore.ctypes.data_as(ctypes.c_char_p), 
		back.ctypes.data_as(ctypes.c_char_p), 
		h, w, d,
		# 浮点数最好显示转换
		ctypes.c_float(mask_threshold)
	)
	return occulusion


occulusion_pos_threshold = 1.5
forward_occulusion_using_pos  = compute_occulusion_using_pos(forward_flow,  backward_flow, mask_threshold=occulusion_pos_threshold)
backward_occulusion_using_pos = compute_occulusion_using_pos(backward_flow, forward_flow,  mask_threshold=occulusion_pos_threshold)
cv_write(add_to_save("forward_occulusion_using_pos.png"),  forward_occulusion_using_pos)
cv_write(add_to_save("backward_occulusion_using_pos.png"), backward_occulusion_using_pos)
cv_show(numpy.concatenate([forward_occulusion_using_pos, backward_occulusion_using_pos], axis=0))
