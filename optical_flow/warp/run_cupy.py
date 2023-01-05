# Python
import os
import sys
import time
# 3rd party
import cv2
import cupy
import numpy
import ctypes
from numpy.ctypeslib import ndpointer


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])



save_dir = "./temp/cupy"
os.makedirs(save_dir, exist_ok=True)


# 先读取图像
image1 = cv2.imread("./images/sintel/frame_0016.png")
image2 = cv2.imread("./images/sintel/frame_0017.png")
height, width, channel = image1.shape
make_show              = True if (height * width < 1024 * 768) else False

# 获取 image1 → image2 的光流
use_flow_cache      = True
save_flow_cache     = True
forward_flow_cache  = "./images/sintel/forward_flow.npy"
backward_flow_cache = "./images/sintel/backward_flow.npy"
if (use_flow_cache):
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
cv_show(numpy.concatenate([forward_flow_visualize, backward_flow_visualize], axis=0))



cuda_source = r'''
extern "C" {
	
	__device__ __forceinline__ float clip_float(const float x, const float low, const float high) {
		if (x < low) return low;
		else if (x > high) return high;
		else return x;
	}

	__device__ __forceinline__ unsigned char clip_uchar(const float x, unsigned char low, unsigned char high) {
		if (x < low) return low;
		else if (x > high) return high;
		else return x;
	}

	__global__ void backward_warp_kernel(
		unsigned char* result,
		unsigned char* source,
		float* flow,
		int height,
		int width,
		int channel
	) {
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		int j = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < height && j < width) {
			float* flow_ptr = flow + 2 * (i * width + j);
			// 获取当前坐标 和 对应的光流值
			float x = i + flow_ptr[1];
			float y = j + flow_ptr[0];
			// 截断
			x = clip_float(x, 0.f, (height - 1) * 1.f);
			y = clip_float(y, 0.f, (width - 1) * 1.f);
			// 上下界限
			const int x_low  = floor(x);
			const int x_high = min(x_low + 1, height - 1);
			const int y_low  = floor(y);
			const int y_high = min(y_low + 1, width - 1);
			// 算加权系数
			const float x_high_weight = x   - x_low;
			const float x_low_weight  = 1.f - x_high_weight;
			const float y_high_weight = y   - y_low;
			const float y_low_weight  = 1.f - y_high_weight;
			// 开始多通道加权
			unsigned char* res_ptr = result + (i * width + j) * channel;
			for (int c = 0; c < channel; ++c) {
				// 找到四个原图四个点的值
				unsigned char Q1 = source[(x_low * width + y_low) * channel + c];
				unsigned char Q2 = source[(x_low * width + y_high) * channel + c];
				unsigned char Q3 = source[(x_high * width + y_low) * channel + c];
				unsigned char Q4 = source[(x_high * width + y_high) * channel + c];
				// 左右加权
				float up_value   = y_low_weight * Q1 + y_high_weight * Q2;
				float down_value = y_low_weight * Q3 + y_high_weight * Q4;
				// 上下加权
				float value = x_low_weight * up_value + x_high_weight * down_value;
				res_ptr[c] = clip_uchar(value, 0, 255);
			}
		}
	}



	__device__ __forceinline__ float compute_flow_intensity(const float u, const float v) {
		return abs(u) + abs(v);
	}


	__global__ void forward_warp_kernel(
		unsigned char* result,
		unsigned char* source,
		float* flow,
		float* flow_intensity,
		unsigned int* lock,
		int height,
		int width,
		int channel
	) {
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		int j = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < height && j < width) {
			float* flow_ptr = flow + i * width * 2;
			// 获取当前坐标 和 对应的光流值
			float u = flow_ptr[2 * j + 1];
			float v = flow_ptr[2 * j];
			// 获取偏移的位置
			float x = clip_float(i + u, 0.f, (height - 1) * 1.f);
			float y = clip_float(j + v, 0.f, (width - 1) * 1.f);
			// 找到 (i + u, j + v 最近的那个点)
			int __x = int(x + 0.5f);
			int __y = int(y + 0.5f);
			
			// 先检查 (__x, __y) 位置, 是不是被赋值过, 如果本次的光流强度更大, 默认是前景, 保留前景
			int target_pos = __x * width + __y;
			float intensity = abs(u) + abs(v);

			// 锁住(这里有 bug, linux 下正常, windows 下死锁)
			while (atomicCAS(lock + target_pos, 0, 1) != 0);

			if (intensity > flow_intensity[target_pos]) {
				// 本次光流更大, 覆盖之前的结果
				flow_intensity[target_pos] = intensity;
				unsigned char* source_start = source + (i * width + j) * channel;
				unsigned char* result_start = result + (__x * width + __y) * channel;
				for (int c = 0; c < channel; ++c) {
					result_start[c] = source_start[c];
				}
			}

			// 释放锁
			lock[target_pos] = 0;
		}
	}
}
'''

# 编译 CUDA 代码
cuda_module = cupy.RawModule(code=cuda_source)
backward_warp_kernel = cuda_module.get_function("backward_warp_kernel")


# 使用 1to2 的光流 forward flow, 将第二帧 warp 到第一帧的位置
# 首先把输入数据传送到 GPU
image2_cuda      = cupy.asarray(image2)
forward_flow_cuda = cupy.asarray(forward_flow)

# 准备一个结果
backward_warp_2to1_cuda = cupy.zeros(image2.shape, dtype="uint8")

# 执行 kernel
CUDA_CEIL = lambda x, y: int((x + y - 1) / y)
backward_warp_kernel(
	(CUDA_CEIL(width, 32), CUDA_CEIL(height, 32)),
	(32, 32),
	(backward_warp_2to1_cuda, image2_cuda, forward_flow_cuda, height, width, channel)
)

# 把数据从 GPU 取到 CPU
backward_warp_2to1 = cupy.asnumpy(backward_warp_2to1_cuda)


# 同理可以得到 1to2 的 backward warp 结果(使用 backward flow)
image1_cuda            = cupy.asarray(image1)
backward_flow_cuda      = cupy.asarray(backward_flow)
backward_warp_1to2_cuda = cupy.zeros(image2.shape, dtype="uint8")
backward_warp_kernel(
	(CUDA_CEIL(width, 32), CUDA_CEIL(height, 32)),
	(32, 32),
	(backward_warp_1to2_cuda, image1_cuda, backward_flow_cuda, height, width, channel)
)
backward_warp_1to2 = cupy.asnumpy(backward_warp_1to2_cuda)
cv2.imwrite(os.path.join(save_dir, "forward_warp_1to2.png"), backward_warp_1to2)
cv_show(backward_warp_1to2)



# forward_warp 很特殊
# 因为需要判断前景后景光流值的大小, 所以需要一个矩阵记录每一个点的光流值(必须初始化 0)
# 同时, 多个线程可以在同一时刻访问同一个点 (x + u, y + v), 所以需要同步
# 第 1 种思路, 使用原子操作！但 CUDA 的原子操作只能支持简单 16/32/64 位数据的加减与或运算这种, 没法支持 uchar、逻辑更复杂的一小段代码
# 第 2 种思路, 使用互斥锁, 每一个点的逻辑判断需要处在临界区

flow_intensity         = cupy.zeros((height, width), dtype="float32")
flow_lock         	   = cupy.zeros((height, width), dtype='uint32') # 注意原子操作不能是 uchar
forward_warp_2to1_cuda = cupy.zeros(image1.shape, dtype="uint8")

# 这一步在 windows 下会卡死??, bug 暂未解决
forward_warp_kernel = cuda_module.get_function("forward_warp_kernel")
forward_warp_kernel(
	(CUDA_CEIL(width, 32), CUDA_CEIL(height, 32)),
	(32, 32),
	(forward_warp_2to1_cuda, image2_cuda, backward_flow_cuda, flow_intensity, flow_lock, height, width, channel)
)

forward_warp_2to1 = cupy.asnumpy(forward_warp_2to1_cuda)
cv_show(forward_warp_2to1)
cv2.imwrite(os.path.join(save_dir, "forward_warp_2to1.png"), forward_warp_2to1)