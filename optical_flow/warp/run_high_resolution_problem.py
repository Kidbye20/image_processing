# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
import ctypes
from numpy.ctypeslib import ndpointer
# https://github.com/xiaofeng94/GMFlowNet/blob/master/core/utils/flow_viz.py
import flow_viz


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y)


save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)
add_to_save = lambda x: os.path.join(save_dir, x)


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
if (use_flow_cache and os.path.exists(forward_flow_cache)):
	forward_flow    = numpy.load(forward_flow_cache)
	backward_flow   = numpy.load(backward_flow_cache)
else:
	forward_flow, backward_flow = flow_viz.compute_optical_flow(image1, image2)

# 如果确认缓存光流, 而且大小不是很大, 缓存之
if (save_flow_cache and make_show):
	numpy.save(forward_flow_cache,  forward_flow)
	numpy.save(backward_flow_cache, backward_flow)


# 可视化光流
forward_flow_visualize  = flow_viz.flow_to_image(forward_flow)[:, :, ::-1] # [:, :, ::-1] 是为了 opencv 显示 BGR 序
backward_flow_visualize = flow_viz.flow_to_image(backward_flow)[:, :, ::-1]
cv_write(add_to_save("forward_flow_visualize_highres.png"),  forward_flow_visualize)
cv_write(add_to_save("backward_flow_visualize_highres.png"), backward_flow_visualize)



# 编译 C++ 代码用于 forward warp
warp_lib_path = "./crane_warp.so"
# 加载动态库
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)
# 对高分辨率图像, 使用上采样的光流做 forward warp
def forward_warp(x, flow):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), dtype="uint8")
	warp_lib.forward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c
	)
	return warped

forward_warp_1to2_highres = forward_warp(image1, forward_flow)
cv_write(add_to_save("forward_warp_1to2_highres.png"), forward_warp_1to2_highres)


def backward_warp(x, flow, mode="bilinear"):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), dtype="uint8")
	warp_lib.backward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p), 
		x.ctypes.data_as(ctypes.c_char_p), 
		flow.ctypes.data_as(ctypes.c_char_p), 
		h, w, c,
		mode.encode()
	)
	return warped

backward_warp_1to2_highres = backward_warp(image1, backward_flow)
cv_write(add_to_save("backward_warp_1to2_highres.png"), backward_warp_1to2_highres)


##########################################################################################
#                       使用最近邻做插值看看效果
##########################################################################################

forward_flow, backward_flow = flow_viz.compute_optical_flow(image1, image2, upsample_mode="nearest")

# 可视化光流
forward_flow_visualize  = flow_viz.flow_to_image(forward_flow)[:, :, ::-1] # [:, :, ::-1] 是为了 opencv 显示 BGR 序
backward_flow_visualize = flow_viz.flow_to_image(backward_flow)[:, :, ::-1]
cv_write(add_to_save("forward_flow_visualize_highres_nearest.png"),  forward_flow_visualize)
cv_write(add_to_save("backward_flow_visualize_highres_nearest.png"), backward_flow_visualize)


# 做 warp
forward_warp_1to2_highres = forward_warp(image1, forward_flow)
cv_write(add_to_save("forward_warp_1to2_highres_nearest.png"), forward_warp_1to2_highres)


backward_warp_1to2_highres = backward_warp(image1, backward_flow)
cv_write(add_to_save("backward_warp_1to2_highres_nearest.png"), backward_warp_1to2_highres)
