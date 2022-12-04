// C && C++
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>


template<typename T, typename A, typename B>
inline T clip(const T x, const A lhs, const B rhs) {
	if (x < lhs) return lhs;
	else if (x > rhs) return rhs;
	else return x;
}


template<const int dimension>
void fast_warp_using_flow_inplementation(
		unsigned char* result, 
		unsigned char* source,
		float* flow,
		int height, 
		int width, 
		int channel) {
	// 遍历每一个位置
	for (int i = 0; i < height; ++i) {
		float* flow_ptr = flow + i * width * dimension;
		for (int j = 0; j < width; ++j) {
			// 获取当前坐标 和 对应的光流值
			float x = i + flow_ptr[2 * j + 1];
			float y = j + flow_ptr[2 * j];
			// 截断
			x = clip(x, 0.f, (height - 1) * 1.f);
			y = clip(y, 0.f, (width - 1)  * 1.f);
			// 上下界限
			const int x_low  = std::floor(x);
			const int x_high = std::min(x_low + 1, height - 1);
			const int y_low  = std::floor(y);
			const int y_high = std::min(y_low + 1, width - 1);
			// 算加权系数
			const float x_high_weight = x - x_low;
			const float x_low_weight  = 1.f - x_high_weight;
			const float y_high_weight = y - y_low;
			const float y_low_weight  = 1.f - y_high_weight;
			// 开始多通道加权
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
				result[(i * width + j) * channel + c] = 
					clip<unsigned char, unsigned char, unsigned char>(value, 0, 255);
			}
		}
	}
}



void fast_compute_occulusion_inplementation(
		unsigned char* occulusion,
		float* forward_flow, 
		float* backward_flow,
		int height, 
		int width,
		int dimension,
		float dis_threshold) {
	// 遍历每一个点
	for (int i = 0; i < height; ++i) {
		float* forward_ptr = forward_flow + i * width * dimension;
		unsigned char* occulusion_ptr = occulusion + i * width;
		for (int j = 0; j < width; ++j) {
			// 获取当前点 (i, j) 经过位移 (u, v), 移动到 (x, y)
			float x = clip(i + forward_ptr[2 * j + 1], 0.f, (height - 1) * 1.f);
			float y = clip(j + forward_ptr[2 * j],     0.f, (width  - 1) * 1.f);
			// 得到上下界限
			int x_low  = std::floor(x);
			int x_high = std::min(x_low + 1, height - 1);
			int y_low  = std::floor(y);
			int y_high = std::min(y_low + 1, width - 1);
			// 得到加权系数
			float x_high_weight = x - x_low;
			float x_low_weight  = 1.f - x_high_weight;
			float y_high_weight = y - y_low;
			float y_low_weight  = 1.f - y_high_weight;
			// 求解 (x, y) 处的反向光流, 插值得到
			float Q1 = backward_flow[(x_low * width + y_low) * dimension];
			float Q2 = backward_flow[(x_low * width + y_high) * dimension];
			float Q3 = backward_flow[(x_high * width + y_low) * dimension];
			float Q4 = backward_flow[(x_high * width + y_high) * dimension];
			float up_value   = y_low_weight * Q1 + y_high_weight * Q2;
			float down_value = y_low_weight * Q3 + y_high_weight * Q4;
			float v  = x_low_weight * up_value + x_high_weight * down_value;
			if (std::abs(y + v - j) > dis_threshold) {
				occulusion_ptr[j] = 255;
				continue;
			}
			Q1 = backward_flow[(x_low * width + y_low) * dimension + 1];
			Q2 = backward_flow[(x_low * width + y_high) * dimension + 1];
			Q3 = backward_flow[(x_high * width + y_low) * dimension + 1];
			Q4 = backward_flow[(x_high * width + y_high) * dimension + 1];
			up_value   = y_low_weight * Q1 + y_high_weight * Q2;
			down_value = y_low_weight * Q3 + y_high_weight * Q4;
			float u  = x_low_weight * up_value + x_high_weight * down_value;
			if (std::abs(x + u - i) > dis_threshold)
				occulusion_ptr[j] = 255;
		}
	}
}



// 编程命名接口
extern "C" {
	void fast_warp_using_flow(
		unsigned char* result, 
		unsigned char* source,
		float* flow,
		int height, 
		int width, 
		int channel) {
		fast_warp_using_flow_inplementation<2>(result, source, flow, height, width, channel);
	}

	void fast_compute_occulusion(
		unsigned char* occulusion,
		float* forward_flow, 
		float* backward_flow,
		int height, 
		int width,
		int dimension,
		float dis_threshold) {
		fast_compute_occulusion_inplementation(occulusion, forward_flow, backward_flow, height, width, dimension, dis_threshold);
	}
}