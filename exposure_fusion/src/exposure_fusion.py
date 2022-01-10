# Python
import os
import sys
import math
import copy
# 3rd party
import cv2
import numpy


def cv_show(image):
	cv2.imshow("crane", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def exposure_fusion(sequence, alphas=(1.0, 1.0, 1.0), best_illumination=0.5, sigma=0.2, eps=1e-12, 
		use_lappyr=True, use_gaussi=False, layers_num=4, scale=2.0):
	# 转化成 float 数据
	sequence = [it.astype("float32") / 255 for it in sequence]
	S = len(sequence)
	H, W, C = sequence[0].shape
	# 准备一些中间变量
	laplace_kernel = numpy.array(([0,  1, 0], [1, -4, 1], [0,  1, 0]), dtype="float32")
	mse = lambda l, r: (l - r) * (l - r)
	best_illumination = numpy.full((H, W), best_illumination, dtype='float32')
	# 存放每张图像的权重图
	weights = []
	# 每张图像都求一个初始权重
	for s in range(S):
		# 从拉普拉斯求对比度
		gray = cv2.cvtColor(sequence[s], cv2.COLOR_BGR2GRAY)
		contrast = cv2.filter2D(gray, -1, laplace_kernel, borderType=cv2.BORDER_REPLICATE)
		contrast = numpy.abs(contrast)
		# 求饱和度
		mean = numpy.mean(sequence[s], axis=-1)
		saturation = numpy.sqrt(numpy.mean([mse(sequence[s][:, :, ch], mean) for ch in range(3)], axis=0))
		# 求亮度
		illumination = [numpy.exp(-0.5 * mse(sequence[s][:, :, ch], best_illumination) / (sigma * sigma)) for ch in range(3)]
		illumination = numpy.prod(illumination, axis=0)
		# 三者加权
		cur_weight = numpy.power(contrast, alphas[0]) * numpy.power(saturation, alphas[1]) * numpy.power(illumination, alphas[2])
		weights.append(cur_weight)
	# 归一化
	weights = numpy.stack(weights, axis=0)
	weights += eps
	weights_sum = numpy.stack([numpy.sum(weights, axis=0) for s in range(S)], axis=0)
	weights /= weights_sum
	# 如果不用 laplace pyramid 融合
	if(not use_lappyr):
		# 对权重图做平滑
		if(use_gaussi): weights = [cv2.GaussianBlur(w, (7, 7), 1.2) for w in weights]
		# 根据这个权重, 直接叠加所有图像
		fusion = numpy.sum([sequence[s] * numpy.tile(numpy.expand_dims(weights[s], axis=-1), (1, 1, 3)) for s in range(S)], axis=0)
		fusion = numpy.clip(fusion * 255, 0, 255).astype("uint8")
	else:
		# 先根据 scale 构造金字塔的每一层大小
		scales = [(W, H)]
		for s in range(1, layers_num): 
			scales.append((math.ceil(scales[s - 1][0] / scale), math.ceil(scales[s - 1][1] / scale)))
		print(scales)
		# 构造金字塔的函数
		def build_gaussi_pyramid(high_res):
			pyramid = []
			for s in range(S):
				this_flash = [high_res[s]]
				for i in range(1, layers_num):
					# 先对当前权重做高斯模糊, 然后下采样
					blurred = cv2.GaussianBlur(this_flash[i - 1], (5, 5), 0.83)
					# blurred = cv2.resize(blurred, scales[i])
					blurred = blurred[::2, ::2]
					this_flash.append(blurred)
					# cv_show(blurred_weight)
				pyramid.append(this_flash)
			return pyramid
		# 先构造 weights 图的高斯金字塔
		weights_pyramid = build_gaussi_pyramid(weights)
		# 构造输入图像的 laplace 金字塔
		images_gaussi_pyramid = build_gaussi_pyramid(sequence)
		images_laplace_pyramid = []
		for s in range(S):
			this_flash = []
			# 从最低分辨率开始向上上采样
			upsampled = copy.deepcopy(images_gaussi_pyramid[s][-1])
			for i in range(layers_num - 2, -1, -1):
				upsampled = cv2.resize(upsampled, scales[i])
				# print(images_gaussi_pyramid[s][i])
				this_flash.append(images_gaussi_pyramid[s][i] - upsampled)
			this_flash.reverse()
			images_laplace_pyramid.append(this_flash)
		# 生成最低分辨率的起始图像
		start = numpy.sum([images_gaussi_pyramid[s][-1] * numpy.stack([weights_pyramid[s][-1] for ch in range(3)], axis=-1) for s in range(S)], axis=0)
		# 根据 weights 和 laplace 金字塔得到融合的拉普拉斯金字塔
		fusion_laplace_pyramid = []
		for i in range(0, layers_num - 1): # 每个尺度, 融合多张图像的 laplace 细节信息, 得到一张图 einops
			# sum 即可
			this_scale = images_laplace_pyramid[0][i] * numpy.stack([weights_pyramid[0][i] for ch in range(3)], axis=-1)
			weight_sum = numpy.sum([numpy.stack([weights_pyramid[s][i] for ch in range(3)], axis=-1) for s in range(S)], axis=0)
			for s in range(1, S):
				this_scale += images_laplace_pyramid[s][i] * numpy.stack([weights_pyramid[s][i] for ch in range(3)], axis=-1)
			this_scale /= weight_sum
			fusion_laplace_pyramid.append(this_scale)
		# 从起始图像, 根据 laplace_pyramid 重构图像
		fusion = start
		for i in range(layers_num - 2, -1, -1):
			fusion = cv2.resize(fusion, scales[i])
			fusion += fusion_laplace_pyramid[i]
		fusion = numpy.clip(fusion * 255, 0, 255).astype('uint8')
	return fusion

# 读取图片
images_dir = "../images/input/2"
images_list = [os.path.join(images_dir, it) for it in os.listdir(images_dir)]
images = [cv2.imread(it) for it in images_list]

fusion_result = exposure_fusion(images, alphas=(1.0, 1.0, 1.0), use_lappyr=True)

# 展示和保存结果
cv_show(fusion_result)
save_dir = "../images/output/2"
cv2.imwrite(os.path.join(save_dir, "laplace_pyramid_fusion.png"), fusion_result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
