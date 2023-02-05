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


# 只转换一次视角, 计算遮挡

# 一个是直接对图做 warp

# 另一种做法是对一个全 1 的 mask 做 warp ===> forward warp 和 backward warp 都写一下