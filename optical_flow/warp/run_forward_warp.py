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

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# 读取两张图象
image1 = cv2.imread("./frame_0016.png")
image2 = cv2.imread("./frame_0017.png")

# 获取 image1 → image2 的光流
use_flow_cache     = False
forward_flow_cache = "./forward_flow.npy"
if (use_flow_cache):
	forward_flow = numpy.load(forward_flow_cache)
else:
	# 从头开始推理获得光流
	import onnxruntime
	onnx_file = "./RAFT-sim.onnx" # GMFlowNet 精度对不齐
	infer_task = onnxruntime.InferenceSession(onnx_file)
	# GMFlowNet 只支持边长为 8 倍数的图像
	height, width, _ = image1.shape
	height_2, width_2 = 8 * (int(height / 8) + 1), 8 * (int(width / 8) + 1)
	h_pad, w_pad = int((height_2 - height) / 2), int((width_2 - width) / 2)
	# 设置转换函数
	def convert_to_tensor(x):
		x_tensor = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
		x_tensor = numpy.pad(x_tensor, [(h_pad, h_pad), (w_pad, w_pad), (0, 0)], mode="reflect")
		x_tensor = x_tensor.transpose(2, 0, 1)
		x_tensor = numpy.ascontiguousarray(x_tensor)
		x_tensor = numpy.expand_dims(x_tensor, axis=0)
		return x_tensor.astype("float32")
	# 把原始图像转换成 B x C x H x W 的内存格式
	image1_tensor = convert_to_tensor(image1)
	image2_tensor = convert_to_tensor(image2)
	print("image1  :  {}\nimage2  :  {}\n".format(image1_tensor.shape, image2_tensor.shape))
	# 开始推理
	[forward_flow] = infer_task.run(["flow"], {"image1": image1_tensor, "image2": image2_tensor})
	forward_flow = numpy.ascontiguousarray(forward_flow[0].transpose(1, 2, 0))
	# 把光流放缩成原始尺度, 注意光流值也要乘
	forward_flow = forward_flow[:height, :width].copy()
	# forward_flow[:, :, 0] *= float(width  / width_2)
	# forward_flow[:, :, 1] *= float(height / height_2)
	# 缓存下光流结果
	numpy.save(forward_flow_cache, forward_flow)
	# 清下内存
	del infer_task, image1_tensor, image2_tensor


# 展示光流
forward_flow_visualize = flow_viz.flow_to_image(forward_flow)[:, :, ::-1]
cv_show(forward_flow_visualize)


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
cv_show(forward_warp_1to2)

# 尝试第一种 warp
forward_warp_1to2_full = forward_warp(image1, forward_flow, mode="full")
cv_write("./results/forward_warp_1to2_full.png", forward_warp_1to2)
cv_show(forward_warp_1to2_full)

# 尝试第二种 warp
forward_warp_1to2_interp = forward_warp(image1, forward_flow, mode="interpolation")
cv_write("./results/forward_warp_1to2_interp.png", forward_warp_1to2)
cv_show(forward_warp_1to2_interp)