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
image1 = cv2.imread("./images/real/IMG_20230118_182911.jpg")
image2 = cv2.imread("./images/real/IMG_20230118_182922.jpg")
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

cv_write("./results/forward_flow_real.png", forward_flow_visualize)

# ==========> 以下尝试做 backward warp, 观察时差交界的地方是否有严重的形变


# 编译 C++ 代码用于 backward warp
warp_lib_path = "./crane_warp.so"
os.system("g++ -fPIC -shared -O2 ./crane_warp.cpp -o {}".format(warp_lib_path))

# 加载动态库
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)

# 简易接口
def backward_warp(x, flow, mode=""):
	h, w, c = x.shape
	warped = numpy.zeros((h, w, c), x.dtype)
	# 如果在 warp 阶段使用插值
	warp_lib.backward_warp_using_flow(
		warped.ctypes.data_as(ctypes.c_char_p),
		x.ctypes.data_as(ctypes.c_char_p),
		flow.ctypes.data_as(ctypes.c_char_p),
		h, w, c,
		# 这个必须加
		mode.encode()
	)
	return warped

# backward warp, 用 image1 -> image2 光流, 把 image2 转换到 image1 视角
backward_warp_2to1 = backward_warp(image2, forward_flow, mode="bilinear")
cv_write("./results/backward_warp_2to1_error.png", backward_warp_2to1)
		


# 试试最近邻
backward_warp_2to1 = backward_warp(image2, forward_flow, mode="nearest")
cv_write("./results/backward_warp_2to1_error_nearest.png", backward_warp_2to1)
