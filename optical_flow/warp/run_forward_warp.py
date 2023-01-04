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
make_show = True if (height * width < 1024 * 768) else False


# 获取 image1 → image2 的光流
use_flow_cache      = False
forward_flow_cache  = "./images/real/forward_flow.npy"
backward_flow_cache = "./images/real/backward_flow.npy"
if (use_flow_cache):
	forward_flow    = numpy.load(forward_flow_cache)
	backward_flow   = numpy.load(backward_flow_cache)
else:
	# 如果是高分辨率的图像
	if (height * width > 1024 * 768):
		make_resize  = True
		height_small = 448
		width_small  = 600
		make_pad     = False
	else:
		make_resize  = False
		# GMFlowNet 只支持边长为 8 倍数的图像, 所以需要做 padding
		if (int(height / 8) == 0 and int(width / 8) == 0): 
			make_pad = True
			height_2, width_2 = 8 * (int(height / 8) + 1), 8 * (int(width / 8) + 1)
			h_pad, w_pad = int((height_2 - height) / 2), int((width_2 - width) / 2)
			print("h_pad  :  {}\nw_pad  :  {}\n".format(h_pad, w_pad))

	# 设置转换函数, 从 numpy、uint8、BGR序、HWC → numpy、float32、RGB序、1CHW
	def convert_to_tensor(x):
		x_tensor = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
		if (make_resize):
			x_tensor = cv2.resize(x_tensor, (width_small, height_small), cv2.INTER_LANCZOS4)
		elif (make_pad):
			x_tensor = numpy.pad(x_tensor, [(h_pad, h_pad), (w_pad, w_pad), (0, 0)], mode="reflect")
		x_tensor = x_tensor.transpose(2, 0, 1)
		x_tensor = numpy.ascontiguousarray(x_tensor)
		x_tensor = numpy.expand_dims(x_tensor, axis=0)
		return x_tensor.astype("float32")

	# 把原始图像转换成 B x C x H x W 的内存格式
	image1_tensor = convert_to_tensor(image1)
	image2_tensor = convert_to_tensor(image2)
	print("image1  :  {}\nimage2  :  {}\n".format(image1_tensor.shape, image2_tensor.shape))

	# 加载光流的 ONNX 模型
	import onnxruntime
	onnx_file  = "GMFlowNet.onnx"
	infer_task = onnxruntime.InferenceSession(onnx_file, 
		providers=['CPUExecutionProvider'])
	# 开始推理
	[forward_flow]  = infer_task.run(["flow"], {"image1": image1_tensor, "image2": image2_tensor})
	[backward_flow] = infer_task.run(["flow"], {"image1": image2_tensor, "image2": image1_tensor})
	forward_flow    = numpy.ascontiguousarray(forward_flow[0].transpose(1, 2, 0))
	backward_flow   = numpy.ascontiguousarray(backward_flow[0].transpose(1, 2, 0))

	# 如果做了放缩, 则需要对图像做放缩, 同时光流要乘以放缩倍率
	if (make_resize):
		forward_flow  = cv2.resize(forward_flow, (width, height), cv2.INTER_LANCZOS4)
		backward_flow = cv2.resize(backward_flow, (width, height), cv2.INTER_LANCZOS4)
		# 倍率
		h_ratio = float(height / forward_flow.shape[0])
		w_ratio = float(width  / forward_flow.shape[1])
		forward_flow[:, :, 0]  *= w_ratio
		forward_flow[:, :, 1]  *= h_ratio
		backward_flow[:, :, 0] *= w_ratio
		backward_flow[:, :, 1] *= h_ratio
	elif (make_pad):
		forward_flow  = forward_flow[h_pad: h_pad + height, w_pad: w_pad + width].copy()
		backward_flow = backward_flow[h_pad: h_pad + height, w_pad: w_pad + width].copy()
	# 缓存下光流结果
	numpy.save(forward_flow_cache, forward_flow)
	numpy.save(backward_flow_cache, backward_flow)
	# 清下内存
	del infer_task, image1_tensor, image2_tensor





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
cv_write("./results/forward_warp_1to2_full.png", forward_warp_1to2)
if (make_show): cv_show(forward_warp_1to2_full)

# 尝试第二种 warp
forward_warp_1to2_interp = forward_warp(image1, forward_flow, mode="interpolation")
cv_write("./results/forward_warp_1to2_interp.png", forward_warp_1to2)
if (make_show): cv_show(forward_warp_1to2_interp)