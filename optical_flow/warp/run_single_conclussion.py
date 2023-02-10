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

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


warp_lib_path = "./crane_warp.so"
# os.system("g++ -fPIC -shared -O2 ./crane_warp.cpp -o {}".format(warp_lib_path))
warp_lib = ctypes.cdll.LoadLibrary(warp_lib_path)
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



# 直接加载之前算好的光流
forward_flow  = numpy.load("./images/sintel/forward_flow.npy")
backward_flow = numpy.load("./images/sintel/backward_flow.npy")

# 读取 image1, image2
image1 = cv2.imread("./images/sintel/frame_0016.png")
image2 = cv2.imread("./images/sintel/frame_0017.png")
height, width, channel = image1.shape

# 使用 image2 → image1 的光流, 把 image1 warp 到 image2
backward_warp_1to2 = backward_warp(image1, backward_flow)



###############################################################################
#                   对图像做 warp, 直接和 image2 比较差值
###############################################################################
diff = numpy.abs(image2.astype("int32") - backward_warp_1to2.astype("int32"))

# 对最后一个通道求均值, 为了展示, 输出 uint8 类型
diff = numpy.mean(diff.reshape((height * width, channel)), axis=-1).reshape((height, width)).astype("uint8")
diff_threshold = 25
mask = ((diff > diff_threshold) * 255).astype("uint8")
cv_show(mask)

# 这种做法，有一个很明显的问题
# 如果是亮度什么的没对齐, 或者说一些 HDR 场景, 直接比较亮度就会有问题



###############################################################################
#                   对一张全 1 mask 做 backward warp, 看看能不能算出遮挡
###############################################################################


def backward_warp_for_mask(flow):
	h, w, _     = flow.shape
	ones_mask   = numpy.full((h, w), 255, dtype="uint8")
	occulussion = numpy.zeros((h, w), dtype="uint8")
	warp_lib.backward_warp_for_mask(
		occulussion.ctypes.data_as(ctypes.c_char_p), 
		ones_mask.ctypes.data_as(ctypes.c_char_p),
		flow.ctypes.data_as(ctypes.c_char_p),
		h, w
	)
	return occulussion

# 这一部分, 用 forward warp 暂时没法算遮挡
occulussion_mask = backward_warp_for_mask(backward_flow)
# cv_show(occulussion_mask)

# 这一部分没法算遮挡, backward warp 除了超出图像范围的内容, 其他的都是满的映射



###############################################################################
#                   对一张全 1 mask 做 backward warp, 看看能不能算出遮挡
###############################################################################

def forward_warp_for_mask(flow):
	h, w, _     = flow.shape
	ones_mask   = numpy.full((h, w), 255, dtype="uint8")
	warp_lib.forward_warp_for_mask(
		ones_mask.ctypes.data_as(ctypes.c_char_p),
		flow.ctypes.data_as(ctypes.c_char_p),
		h, w
	)
	return ones_mask

occulussion_mask = forward_warp_for_mask(forward_flow)
cv_show(occulussion_mask)
