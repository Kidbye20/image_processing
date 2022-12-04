// self
#include "cuda_utils.h"



// 这个 batch_size 是多余的, 还不知道要如何处理
__global__ void fast_lrcheck_kernel(
		unsigned char* occulusion,
		const float* forward_flow,
		const float* backward_flow,
		int batch_size,
		int channel,
		int height,
		int width,
		float dis_threshold) {
	// 先获取坐标
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 检查边界
    if (i < height && j < width) {
    	// 先写一些局部变量
		int channel_len = height * width;

    	// 获取 flow 值, u 跟 v 两个方向
    	float x = i + forward_flow[channel_len + i * width + j];
    	float y = j + forward_flow[i * width + j];
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
		// 求解 (x, y) 处的反向光流, 插值得到
		float Q1 = backward_flow[x_low * width + y_low];
		float Q2 = backward_flow[x_low * width + y_high];
		float Q3 = backward_flow[x_high * width + y_low];
		float Q4 = backward_flow[x_high * width + y_high];
		float up_value   = y_low_weight * Q1 + y_high_weight * Q2;
		float down_value = y_low_weight * Q3 + y_high_weight * Q4;
		float v  = x_low_weight * up_value + x_high_weight * down_value;
		// 这里可能有 bug, 如果初始化不是 0, 就会有问题
		if (abs(y + v - j) > dis_threshold) {
			occulusion[i * width + j] = 255;
			return;
		}
		backward_flow += channel_len;
		Q1 = backward_flow[x_low * width + y_low];
		Q2 = backward_flow[x_low * width + y_high];
		Q3 = backward_flow[x_high * width + y_low];
		Q4 = backward_flow[x_high * width + y_high];
		up_value   = y_low_weight * Q1 + y_high_weight * Q2;
		down_value = y_low_weight * Q3 + y_high_weight * Q4;
		float u  = x_low_weight * up_value + x_high_weight * down_value;
		if (abs(x + u - i) > dis_threshold)
			occulusion[i * width + j] = 255;
    }
}






void optical_flow_lrcheck_cuda(
		unsigned char* occulusion, 
		const float* forward_flow, 
		const float* backward_flow, 
		const int batch_size, 
		const int channel,
		const int height, 
		const int width, 
		const float dis_threshold) {
	// 设置 grid, block
	dim3 block(32, 32);
	dim3 grid(CUDA_CEIL(width, block.x), CUDA_CEIL(height, block.y));

	// 启动 kernel
	fast_lrcheck_kernel<<<grid, block>>>(
		occulusion, 
		forward_flow,
		backward_flow,
		batch_size,
		channel,
		height, 
		width, 
		dis_threshold
	);
}


