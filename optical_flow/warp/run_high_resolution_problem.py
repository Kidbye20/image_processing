# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
import ctypes
import dill as pickle
from numpy.ctypeslib import ndpointer
# https://github.com/xiaofeng94/GMFlowNet/blob/master/core/utils/flow_viz.py
import flow_viz





########################################## 【1】 准备会用到的一些函数 ###########################################
def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y)


# 编译 C++ 代码用于 forward warp
warp_lib_path = "./crane_warp.so"
# 加载动态库
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)
# 对高分辨率图像, 使用上采样的光流做 forward warp
def forward_warp(x, flow):
	h, w, c = x.shape
	assert x.shape[:2] == flow.shape[:2], "the shapes of image and flow must be the same"
	warped = numpy.zeros((h, w, c), dtype="uint8")
	warp_lib.forward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c
	)
	return warped


def backward_warp(x, flow, mode="bilinear"):
	h, w, c = x.shape
	assert x.shape[:2] == flow.shape[:2], "the shapes of image and flow must be the same"
	warped = numpy.zeros((h, w, c), dtype="uint8")
	warp_lib.backward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c,
		mode.encode()
	)
	return warped






########################################## 【2】 准备数据读写 ###########################################
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)
add_to_save = lambda x, y: cv_write(os.path.join(save_dir, x), y)


# 先读取图像
image1 = cv2.imread("./images/real/IMG_20230205_162425.jpg")
image2 = cv2.imread("./images/real/IMG_20230205_162429.jpg")
height, width, channel = image1.shape
make_show              = True if (height * width < 1024 * 768) else False

# 获取 image1 → image2 的光流
use_flow_cache      = True
save_flow_cache     = True
forward_flow_cache  = "./images/real/forward_flow.npy"
backward_flow_cache = "./images/real/backward_flow.npy"
if (use_flow_cache and os.path.exists(forward_flow_cache) and os.path.exists(backward_flow_cache)):
	lowres_forward_flow  = numpy.load(forward_flow_cache)
	lowres_backward_flow = numpy.load(backward_flow_cache)
	lowres_h, lowres_w   = lowres_forward_flow.shape[:2]
	lowres_image1        = cv2.resize(image1, (lowres_w, lowres_h))
	lowres_image2        = cv2.resize(image2, (lowres_w, lowres_h))
else:
	# 这里先不做上采样
	lowres_forward_flow, lowres_backward_flow, lowres_image1, lowres_image2 = \
		flow_viz.compute_optical_flow(image1, image2, do_upsample=False, return_img=True)

# 如果确认缓存光流, 而且大小不是很大, 缓存之
if (save_flow_cache):
	numpy.save(forward_flow_cache,  lowres_forward_flow)
	numpy.save(backward_flow_cache, lowres_backward_flow)


########################################## 【3】 小分辨率上测试 warp ###########################################
print(lowres_forward_flow.shape, lowres_backward_flow.shape, lowres_image1.shape, lowres_image2.shape)

# 首先得到小分辨率的光流, 可视化
forward_flow_visualize  = flow_viz.flow_to_image(lowres_forward_flow,  convert_to_bgr=True) # [:, :, ::-1] 是为了 opencv 显示 BGR 序
backward_flow_visualize = flow_viz.flow_to_image(lowres_backward_flow, convert_to_bgr=True)
cv_show(forward_flow_visualize)
cv_show(backward_flow_visualize)
# 保存小分辨率光流的可视化结果
add_to_save("test_lowres_forward_flow.png",  forward_flow_visualize)
add_to_save("test_lowres_backward_flow.png", backward_flow_visualize)


# 在小分辨率上将 image1 warp 到 image2 视角
# 1. 使用 image1 → image2 的光流, 做 forward warp, 把 image1 移动到 image2
lowres_forward_warp_1to2 = forward_warp(lowres_image1, lowres_forward_flow)
# 2. 使用 image2 → image1 的光流, 做 backward warp, 把 image1 移动到 image2
lowres_backward_warp_1to2 = backward_warp(lowres_image1, lowres_backward_flow)
cv_show(lowres_forward_warp_1to2)
cv_show(lowres_backward_warp_1to2)
add_to_save("test_lowres_forward_warp_1to2.png",  lowres_forward_warp_1to2)
add_to_save("test_lowres_backward_warp_1to2.png", lowres_backward_warp_1to2)




########################################## 【3】 高分辨率上测试 warp, 双线性插值 ###########################################
# 1. 首先把两个光流都上采样到高分辨率, 默认用 bilinear
highres_bilinear_forward_flow  = cv2.resize(lowres_forward_flow,  (width, height), cv2.INTER_LINEAR)
highres_bilinear_backward_flow = cv2.resize(lowres_backward_flow, (width, height), cv2.INTER_LINEAR)
# 更大尺寸上, 要
height_ratio = height / float(lowres_image1.shape[0])
width_ratio  = width  / float(lowres_image1.shape[1])
highres_bilinear_forward_flow[:, :, 1]  *= height_ratio
highres_bilinear_forward_flow[:, :, 0]  *= width_ratio
highres_bilinear_backward_flow[:, :, 1] *= height_ratio
highres_bilinear_backward_flow[:, :, 0] *= width_ratio
# 保存可视化结果
add_to_save("test_highres_bilinear_forward_flow.png",  flow_viz.flow_to_image(highres_bilinear_forward_flow,  convert_to_bgr=True))
add_to_save("test_highres_bilinear_backward_flow.png", flow_viz.flow_to_image(highres_bilinear_backward_flow, convert_to_bgr=True))

# 2. 使用双线性插值上采样的光流做一次
highres_bilinear_forward_warp_1to2  = forward_warp(image1,  highres_bilinear_forward_flow)
highres_bilinear_backward_warp_1to2 = backward_warp(image1, highres_bilinear_backward_flow)
# 保存 warp 结果
add_to_save("test_highres_bilinear_forward_warp_1to2.png",  highres_bilinear_forward_warp_1to2)
add_to_save("test_highres_bilinear_backward_warp_1to2.png", highres_bilinear_backward_warp_1to2)





########################################## 【4】 高分辨率上测试 warp, 最近邻插值 ###########################################
highres_nearest_forward_flow  = cv2.resize(lowres_forward_flow,  (width, height), cv2.INTER_CUBIC)
highres_nearest_backward_flow = cv2.resize(lowres_backward_flow, (width, height), cv2.INTER_CUBIC)
# 更大尺寸上, 要
print("height_ratio  ", height_ratio, "\nweight_ratio  ", width_ratio)
highres_nearest_forward_flow[:, :, 1]  *= height_ratio
highres_nearest_forward_flow[:, :, 0]  *= width_ratio
highres_nearest_backward_flow[:, :, 1] *= height_ratio
highres_nearest_backward_flow[:, :, 0] *= width_ratio

# 为什么最近邻和 bilinear 的差距是 0 ??????
compute_mse = lambda x, y: numpy.abs(x - y).mean()
print("{:.5f}".format(compute_mse(highres_nearest_forward_flow, highres_bilinear_forward_flow)))
print("{:.5f}".format(compute_mse(highres_nearest_backward_flow, highres_bilinear_backward_flow)))

# 保存可视化结果
add_to_save("test_highres_nearest_forward_flow.png",  flow_viz.flow_to_image(highres_nearest_forward_flow,  convert_to_bgr=True))
add_to_save("test_highres_nearest_backward_flow.png", flow_viz.flow_to_image(highres_nearest_backward_flow, convert_to_bgr=True))

# 2. 使用双线性插值上采样的光流做一次
highres_nearest_forward_warp_1to2  = forward_warp(image1,  highres_nearest_forward_flow)
highres_nearest_backward_warp_1to2 = backward_warp(image1, highres_nearest_backward_flow)
# 保存 warp 结果
add_to_save("test_highres_nearest_forward_warp_1to2.png",  highres_nearest_forward_warp_1to2)
add_to_save("test_highres_nearest_backward_warp_1to2.png", highres_nearest_backward_warp_1to2)
