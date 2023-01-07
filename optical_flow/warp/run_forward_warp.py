# Python
import os
import sys
import time
# 3rd party
import cv2
import numpy
import ctypes
from numpy.ctypeslib import ndpointer
# self
import flow_viz


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y) # , [cv2.IMWRITE_PNG_COMPRESSION, 0]



# 读取两张图象
image1 = cv2.imread("./images/real/IMG_20230104_185838.jpg")
image2 = cv2.imread("./images/real/IMG_20230104_185907.jpg")
# 获取图像的形状
height, width, _ = image1.shape
make_show        = True if (height * width < 1024 * 768) else False


# 获取 image1 → image2 的光流
use_flow_cache      = False
save_flow_cache     = False
forward_flow_cache  = "./images/real/forward_flow.npy"
backward_flow_cache = "./images/real/backward_flow.npy"
if (use_flow_cache):
	forward_flow    = numpy.load(forward_flow_cache)
	backward_flow   = numpy.load(backward_flow_cache)
else:
	forward_flow, backward_flow = flow_viz.compute_optical_flow(image1, image2)

# 如果确认缓存光流, 而且大小不是很大, 
if (save_flow_cache and make_show):
	numpy.save(forward_flow_cache,  forward_flow)
	numpy.save(backward_flow_cache, backward_flow)



# 展示光流
forward_flow_visualize  = flow_viz.flow_to_image(forward_flow)[:, :, ::-1]
backward_flow_visualize = flow_viz.flow_to_image(backward_flow)[:, :, ::-1]
if (make_show):
	cv_show(forward_flow_visualize)
	cv_show(backward_flow_visualize)
cv_write("./results/forward_flow_visualize.png",  forward_flow_visualize)
cv_write("./results/backward_flow_visualize.png", backward_flow_visualize)


# 编译 C++ 代码用于 forward warp
warp_lib_path = "./crane_warp.so"
os.system("g++ -fPIC -shared -O2 ./crane_warp.cpp -o {}".format(warp_lib_path))

# 加载动态库
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)

# 简易接口
def forward_warp(x, flow, mode=""):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), x.dtype)
	# 如果在 warp 阶段使用插值
	if (mode == "full"):
		warp_lib.full_forward_warp_using_flow(
			warped.ctypes.data_as(ctypes.c_char_p),
			x.ctypes.data_as(ctypes.c_char_p),
			flow.ctypes.data_as(ctypes.c_char_p),
			h, w, c, 1
		)
	# 如果在 warp 之后做填补
	elif (mode == 'interpolation'):
		warp_lib.interp_forward_warp_using_flow(
			warped.ctypes.data_as(ctypes.c_char_p),
			x.ctypes.data_as(ctypes.c_char_p),
			flow.ctypes.data_as(ctypes.c_char_p),
			h, w, c, 2
		)
	else:
		# 普通模式, 会出现孔洞
		warp_lib.forward_warp_using_flow(
			warped.ctypes.data_as(ctypes.c_char_p),
			x.ctypes.data_as(ctypes.c_char_p),
			flow.ctypes.data_as(ctypes.c_char_p),
			h, w, c
		)
	return warped


# 使用 forward warp 把 image1 变换到 image2 的视角
forward_warp_1to2 = forward_warp(image1, forward_flow)
cv_write("./results/forward_warp_1to2.png", forward_warp_1to2)
if (make_show): cv_show(forward_warp_1to2)

# 尝试第一种 warp
forward_warp_1to2_full = forward_warp(image1, forward_flow, mode="full")
cv_write("./results/forward_warp_1to2_full.png", forward_warp_1to2_full)
if (make_show): cv_show(forward_warp_1to2_full)

# 尝试第二种 warp
forward_warp_1to2_interp = forward_warp(image1, forward_flow, mode="interpolation")
cv_write("./results/forward_warp_1to2_interp.png", forward_warp_1to2_interp)
if (make_show): cv_show(forward_warp_1to2_interp)