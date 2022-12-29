// C++
#include <cmath>
#include <vector>
#include <iostream>



namespace {
	template<typename T>
	inline T square(const T x) {
		return x * x;
	}
}



void joint_bilateral_upsampling_inplementation(
		float* result,
		float* source,
		unsigned char* guide,
		float* extra_args) {
	// 首先解析各项参数
	int args_cnt         = 0;
	int h_small 	     = int(extra_args[args_cnt++]);
	int w_small          = int(extra_args[args_cnt++]);
	int channel_small    = int(extra_args[args_cnt++]);
	int h_large          = int(extra_args[args_cnt++]);
	int w_large 	     = int(extra_args[args_cnt++]);
	int channel_large    = int(extra_args[args_cnt++]);
	int h_small_radius   = int(extra_args[args_cnt++]);
	int w_small_radius   = int(extra_args[args_cnt++]);
	int h_large_radius   = int(extra_args[args_cnt++]);
	int w_large_radius   = int(extra_args[args_cnt++]);
	float h_scale        = extra_args[args_cnt++];
	float w_scale        = extra_args[args_cnt++];
	float spatial_sigma  = extra_args[args_cnt++];
	float range_sigma    = extra_args[args_cnt++];
	bool use_bilinear    = extra_args[args_cnt++] > 0;
	bool use_spatial_lut = extra_args[args_cnt++] > 0;
	bool use_range_lut   = extra_args[args_cnt++] > 0;
	printf("[%d %d %d] [%d %d %d] [%d %d %d %d] [%f %f] [%f %f] [%d %d %d]\n",
		h_small, w_small, channel_small,
		h_large, w_large, channel_large,
		h_small_radius, w_small_radius, h_large_radius, w_large_radius,
		h_scale, w_scale,
		spatial_sigma, range_sigma,
		use_bilinear, use_spatial_lut, use_range_lut);

	// 生成一些辅助变量
	int w_small_padded = w_small + 2 * w_small_radius;
	int w_large_padded = w_large + 2 * w_large_radius;
	printf("[%d %d]\n", w_small_padded, w_large_padded);

	// 提前计算一个空间权重表
	int spatial_size = (2 * h_large_radius + 1) * (2 * w_large_radius + 1);
	std::vector<float> spatial_table;
	if (use_spatial_lut) {
		// 如果使用表优化, 就加速
		spatial_table.resize(spatial_size);
		int spatial_cnt{0};
		for (int x = -h_large_radius; x <= h_large_radius; ++x) {
			for (int y = -w_large_radius; y <= w_large_radius; ++y) {
				spatial_table[spatial_cnt++] = std::exp(-(square<float>(x / h_scale) + square<float>(w_scale)) / (2 * square<float>(spatial_sigma)));
			}
		}
	}

	// 提前计算一个值域权重表, 只有 guide 是 int 族时才能这么干
	constexpr int range_size{3 * 256 * 256};
	constexpr float range_norm{256.f};
	const float range_norm_2 = square(range_norm);
	std::vector<float> range_table;
	if (use_range_lut) {
		// 如果对 range 做表优化
		range_table.resize(range_size);
		for (int i = 0; i < range_size; ++i) {
			range_table[i] = std::exp(-square(i / range_norm_2) / (2 * square(range_sigma)));
		}
	}
		
	// 生成高分辨率结果的每一个点的值
	for (int i = 0; i < h_large; ++i) {
		// 当前行的引导指针
		unsigned char* guide_ptr = guide + channel_large * ((h_large_radius + i) * w_large_padded + w_large_radius);
		// 当前行的结果指针
		float* res_ptr = result + channel_small * i * w_large;
		// 当前行映射到小分辨率所在行
		int   i_small   = int(i / h_scale);
		float* src_ptr = source + channel_small * ((h_small_radius + i_small) * w_small_padded + w_small_radius);
		// 遍历当前行的所有位置
		for (int j = 0; j < w_large; ++j) {
			// 首先找到当前位置在引导图上的值 P
			unsigned char* P = guide_ptr + j * channel_large;
			// 初始化累加值
			float temp[channel_small] = {0.f};
			float weight_sum{0.f};
			// 遍历邻域
			int spatial_cnt{0};
			for (int x = -h_large_radius; x <= h_large_radius; ++x) {
				for (int y = -w_large_radius; y <= w_large_radius; ++y) {
					
					// 查表获取这个位置的空间权重
					float spatial_weight;
					if (use_spatial_lut) 
						spatial_weight = spatial_table[spatial_cnt++];
					else
						spatial_weight = std::exp(-(square<float>(x / h_scale) + square<float>(w_scale)) / (2 * square<float>(spatial_sigma)));
					
					// 获取处在 (i + x, j + y) 的邻域像素
					unsigned char* Q = P + channel_large * (x * w_large_padded + y);
					// 查表获取这个位置的值域差权重
					float range_weight;
					if (use_range_lut) {
						int diff_pos = 0;
						for (int c = 0; c < channel_large; ++c) {
							diff_pos += square<unsigned int>(P[c] - Q[c]);
						}
						range_weight = range_table[diff_pos];
					} else {
						// 这里还有点 bug
						float diff{0.f};
						for (int c = 0; c < channel_large; ++c) {
							diff += square<float>(P[c] / range_norm - Q[c] / range_norm);
						}
						float range_weight = std::exp(-square(diff) / (2 * square(range_sigma)));
					}
						
					// 获取邻域点 (i + x, j + y) 对中心点 (i, j) 的加权值
					float this_weight = spatial_weight * range_weight;
					weight_sum        += this_weight;
					// 找到 (i + x, j + y) 对应在小分辨率的值, 直接加权
					if (not use_bilinear) {
						// 如果不用插值, 直接最近邻获取小分辨率的值
						float* S = src_ptr + channel_small * (int(x / h_scale) * w_small_padded + int((j + y) / w_scale));
						for (int c = 0; c < channel_small; ++c) {
							temp[c] += this_weight * S[c];
						}
					}
					else {
						// 根据相对坐标 (x / h_scale, (j + y) / w_scale) 插值
						float S[channel_small];
						float x_offset = x / h_scale;
						float y_offset = (j + y) / w_scale;
						// 获取上下界
						int x_low  = std::floor(x_offset);
						int x_high = x_low + 1;
						int y_low  = std::floor(y_offset);
						int y_high = y_low + 1;
						// 获取四个坐标位置(用于插值的)
						float* Q1 = src_ptr + channel_small * (x_low * w_small_padded + y_low);
						float* Q2 = Q1 + channel_small;
						float* Q3 = Q1 + channel_small * w_small_padded;
						float* Q4 = Q3 + channel_small;
						// 计算加权值
						float x_high_weight = x_offset - x_low;
						float y_high_weight = y_offset - y_low;
						// 开始加权
						for (int c = 0; c < channel_small; ++c) {
							float up   = (1.f - y_high_weight) * Q1[c] + y_high_weight * Q2[c];
							float down = (1.f - y_high_weight) * Q3[c] + y_high_weight * Q4[c];
							float val  = (1.f - x_high_weight) * up + x_high_weight * down;
							temp[c] += this_weight * val;
						}
					}
				}
			}
			// 赋值
			for (int c = 0; c < channel_small; ++c) {
				res_ptr[channel_small * j + c] = temp[c] / weight_sum;
			}
		}
	}
}

extern "C" {
	void joint_bilateral_upsampling(
			float* result,
			float* source,
			unsigned char* guide,
			float* extra_args) {
		joint_bilateral_upsampling_inplementation(result, source, guide, extra_args);
	}
}