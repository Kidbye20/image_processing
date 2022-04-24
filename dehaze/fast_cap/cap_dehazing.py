# Python
import os
import sys
import math
# 3rd party
import cv2
import numpy


def cv_show(image, message='crane'):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def normalize(image):
	min_value = image.min()
	dynamic = image.max() - min_value
	image = (image - min_value) / dynamic
	return numpy.clip(image * 255, 0, 255).astype("uint8")


def heatmap_show(image):
	heat_map = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)


def evaluate_depth_map(haze_one, thetas=[0.1893, 1.0267, -1.2966]):
	# 转成 hsv 空间
	hsv = cv2.cvtColor(haze_one, cv2.COLOR_BGR2HSV)
	return thetas[0] + thetas[1] * hsv[:, :, 2] + thetas[2] * hsv[:, :, 1]


def evaluate_dark_region(depth, radius=7, use_ctypes=True):
	reflect = cv2.copyMakeBorder(depth, radius, radius, radius, radius, cv2.BORDER_REFLECT)
	# Pybind11
	if(use_ctypes == False):
		import crane 
		return crane.compute_dark(reflect, radius)
	# Ctypes
	else:
		import ctypes
		lib = ctypes.cdll.LoadLibrary("./cpp_extensions/ctypes/example.so")
		pad_ptr = reflect.ctypes.data_as(ctypes.c_char_p)
		rows, cols = reflect.shape
		result = numpy.zeros((rows - 2 * radius, cols - 2 * radius)).astype("float32")
		res_ptr = result.ctypes.data_as(ctypes.c_char_p)
		lib.compute_dark(pad_ptr, res_ptr, rows, cols, radius)
		return result
	# H, W = depth.shape[:2]
	# depth_min = depth.copy()
	# for i in range(H):
	# 	for j in range(W):
	# 		min_value = reflect[i: i + 2 * radius, j: j + 2 * radius].min()
	# 		depth_min[i: i + 2 * radius, j: j + 2 * radius] = \
	# 			numpy.clip(depth_min[i: i + 2 * radius, j: j + 2 * radius], -100, min_value)
	# return depth_min
	


def evaluate_atmospheric_light(haze, depth, radius=7, proportion=0.001, refine=True):
	global history
	# 是否要对深度图做修正 ?
	if(refine):
		# 先找到每个点局部区域的最小值, 记录下来
		depth_region = evaluate_dark_region(depth, radius)
		depth = cv2.ximgproc.guidedFilter(haze, depth_region, 7, 1e-2)
		history["去除一些近景白色得到的深度图"] = heatmap_show(depth_region)
		history["引导滤波精修得到的深度图"] = heatmap_show(depth)

	# 在深度图中, 找最亮的 0.1% 的点坐标
	H, W = depth.shape[:2]
	num = math.ceil(H * W * proportion)
	depth_map_dark_f = depth.flatten()
	brightest = numpy.argsort(depth_map_dark_f)
	brightest = brightest[-num:]
	brightest = (numpy.array([it / W for it in brightest], dtype=numpy.int64), \
				 numpy.array([it % W for it in brightest], dtype=numpy.int64))

	# 画出用于估计 A 的是哪些像素
	display = haze.copy()
	points_num = len(brightest[0])
	for i in range(points_num):
		cv2.circle(display, (brightest[1][i], brightest[0][i]), 1, (0, 0, 255))
	history["用于估计大气光 A 的像素"] = display

	# 对应输入的有雾图像中去找 r, g, b 三通道的最大值, 作为全局大气光的估计
	brightest_pixels = haze[brightest]
	A = brightest_pixels.max(axis=0)

	return A[::-1]


# 记录中间结果
history = {}

# 读取图像
image_path = './images/input/tiananmen1.bmp'
haze_image = cv2.imread(image_path)
I = haze_image.astype("float32") / 255

# 求深度图
beta = 0.8
depth_map = evaluate_depth_map(I, thetas=[0.1893, 1.0267, -1.2966])
history["深度估计图 d"] = heatmap_show(depth_map)

# 根据深度图求解传输率 t
transmission_map = numpy.exp(-beta * depth_map)
t = numpy.clip(transmission_map, 0.05, 1.0)
history["传输率 t"] = heatmap_show(t)

# 根据深度图求解全局大气光 A
A = evaluate_atmospheric_light(I, depth_map, refine=True, radius=7)
print("全局大气光 A  ", A)

# 已知 A, t 求解 J
t = numpy.atleast_3d(t)
J = (I - A) / t + A
J = numpy.clip(J * 255, 0, 255).astype("uint8")
cv_show(numpy.concatenate([haze_image, J], axis=1))

# 展示细节
history["去雾结果"] = J
for l, r in history.items():
	cv_show(r, l)