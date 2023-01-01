// self
#include "cuda_utils.h"





// 这个 batch_size 是多余的, 还不知道要如何处理
__global__ void fast_warp_kernel(
		unsigned char* result,
		const unsigned char* image,
		const float* flow,
		int batch_size,
		int channel,
		int height,
		int width) {
	// 先获取坐标
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 检查边界
    if (i < height && j < width) {
    	// 先写一些局部变量
		int channel_len = height * width;

    	// 获取 flow 值, u 跟 v 两个方向
    	float x = i + flow[channel_len + i * width + j];
    	float y = j + flow[i * width + j];
    	// 截断
    	x = float_clip(x, 0.f, (height - 1) * 1.f);
    	y = float_clip(y, 0.f, (width  - 1) * 1.f);
    	// 上下界限
		int x_low  = floor(x);
		int x_high = min(x_low + 1, height - 1);
		int y_low  = floor(y);
		int y_high = min(y_low + 1, width - 1);
		// 算加权系数
		float x_high_weight = x - x_low;
		float x_low_weight  = 1.f - x_high_weight;
		float y_high_weight = y - y_low;
		float y_low_weight  = 1.f - y_high_weight;
		// 开始多通道加权
		for (int c = 0; c < channel; ++c) {
			// 找到四个原图四个点的值
			unsigned char Q1 = image[x_low * width + y_low];
			unsigned char Q2 = image[x_low * width + y_high];
			unsigned char Q3 = image[x_high * width + y_low];
			unsigned char Q4 = image[x_high * width + y_high];
			// 左右加权
			float up_value   = y_low_weight * Q1 + y_high_weight * Q2;
			float down_value = y_low_weight * Q3 + y_high_weight * Q4;
			// 上下加权
			float value = x_low_weight * up_value + x_high_weight * down_value;
			result[i * width + j] = uint8_clip(value, 0, 255);
			// 偏移
			image  += channel_len;
			result += channel_len;
		}
    }
}



void optical_flow_warp_cuda(
		unsigned char* result, 
		const unsigned char* image, 
		const float* flow, 
		const int batch_size, 
		const int channel,
		const int height, 
		const int width) {
	// 1, 3, 440, 1024
	// 设置 grid, block
	dim3 block(32, 32);
	dim3 grid(CUDA_CEIL(width, block.x), CUDA_CEIL(height, block.y));
	// 启动 kernel
	fast_warp_kernel<<<grid, block>>>(
		result, 
		image,
		flow,
		batch_size,
		channel,
		height, 
		width
	);
}