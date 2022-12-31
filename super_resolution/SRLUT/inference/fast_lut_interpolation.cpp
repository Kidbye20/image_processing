// C & C++
#include <list>
#include <iostream>
#include <random>
#include <assert.h>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>


// 截断
template<typename F=float, typename T=unsigned char>
inline T clip(const F x, const T low, const T high) {
	if (x < low) return low;
	else if (x > high) return high;
	else return x;
}


// 边长 INTERVAL + 1xINTERVAL + 1xINTERVAL + 1xINTERVAL + 1, x4
template<const int L=17, const int D=4>
inline float* acquire(float* LUT, int a, int b, int c, int d) {
	return LUT + D * (a * L * L * L + b * L * L + c * L + d);
}


// 四舍五入
template<typename T>
inline T nearest_round(const float x, const float eps=0.5f) {
	return T(x + eps);
}


template<const int INTERVAL=16, const int SCALE=4>
void fast_lut_interpolation_inplementation(
		unsigned char* result,
		unsigned char* image_padded,
		float* LUT,
		int height,
		int width, 
		int channel,
		int radius) {
	// 3, height, width
	const int height_2 = height + 2 * radius;
	const int width_2  = width  + 2 * radius;
	// 大分辨率图像的边长
	const int new_width = 2 * width;
	// 每个通道分开做
	for (int c = 0; c < channel; ++c) {
		// 遍历当前通道每一个位置
		for (int i = 0; i < height; ++i) {
			unsigned char* image_ptr = image_padded + (radius + i) * width_2 + radius;
			for (int j = 0; j < width; ++j) {
				// 集成结果, 应该会有 4 个值
				float value00{0.f}, value01{0.f}, value10{0.f}, value11{0.f};
				// 找到当前像素周围的 9 个像素
				int A, B, C, D;
				// 遍历四种情况
				for (int t = 0; t < 4; ++t) {
					switch (t) {
					case 0:
						A = image_ptr[j], B = image_ptr[j + 1],        C = image_ptr[j + width_2], D = image_ptr[j + width_2 + 1];
						break;
					case 1: 
						A = image_ptr[j], B = image_ptr[j + width_2],  C = image_ptr[j - 1],       D = image_ptr[j + width_2 - 1];
						break;
					case 2:
						A = image_ptr[j], B = image_ptr[j - 1],        C = image_ptr[j - width_2], D = image_ptr[j - width_2 - 1];
						break;
					case 3:
						A = image_ptr[j], B = image_ptr[j  - width_2], C = image_ptr[j + 1],       D = image_ptr[j - width_2 + 1];
						break;
					default: 
						break;
					}
					// 得到当前的索引 [A][B][C][D], 得先除以 INTERVAL(更改顺序并没有优化)
					int A_low  = A / INTERVAL;
					int B_low  = B / INTERVAL;
					int C_low  = C / INTERVAL;
					int D_low  = D / INTERVAL;
					int A_high = A_low + 1;
					int B_high = B_low + 1;
					int C_high = C_low + 1;
					int D_high = D_low + 1;
					// 得到 LUT 的 INTERVAL 个顶点值(这部分去掉, 0.049s →  0.043s)
					// 求坐标的余数, 从而决定选哪 5 个顶点
					int fa = A % INTERVAL;
					int fb = B % INTERVAL;
					int fc = C % INTERVAL;
					int fd = D % INTERVAL;
					// 候选加权点
					constexpr int CANDIDATE = 5;
					float  weight[CANDIDATE];
					float* points[CANDIDATE];
					// 开始判断
					if (fa > fb) {
						if (fb > fc) {
							if (fc > fd) {
								weight[0] = INTERVAL - fa;  points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fb;        points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fb - fc; 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_low); 
								weight[3] = fc - fd; 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low); 
								weight[4] = fd;      		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
								
							} 
							else if (fb > fd) {
								weight[0] = INTERVAL - fa; points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fb; 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fb - fd; 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_low); 
								weight[3] = fd - fc; 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
								
							}
							else if (fa > fd) {
								weight[0] = INTERVAL - fa; points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fd; 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fd - fb; 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_high); 
								weight[3] = fb - fc; 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
								
							}
							else {
								weight[0] = INTERVAL - fd; points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fd - fa; 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high); 
								weight[2] = fa - fb; 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_high); 
								weight[3] = fb - fc; 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
								
							}
						}
						else if (fa > fc) {
							if (fb > fd) {
					            weight[0] = (INTERVAL - fa); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
					            weight[1] = (fa - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low);
					            weight[2] = (fc - fb); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_low);
					            weight[3] = (fb - fd); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low);
					            weight[4] = fd ;       		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fc > fd) {
								weight[0] = (INTERVAL - fa); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low); 
		                        weight[1] = (fa - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low);
		                        weight[2] = (fc - fd); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_low);
		                        weight[3] = (fd - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
		                        weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							} 
							else if (fa > fd) {
								weight[0] = (INTERVAL - fa); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fa - fd); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_low);
								weight[2] = (fd - fc); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_high);
								weight[3] = (fc - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else {
								weight[0] = (INTERVAL - fd); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fd - fa); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high);
								weight[2] = (fa - fc); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_low, D_high);
								weight[3] = (fc - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
						}
						else {
							if (fb > fd) {
								weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fa); points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fa - fb); points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_low);
								weight[3] = (fb - fd); points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low);
								weight[4] = fd;        points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fc > fd) {
								weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fa); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fa - fd); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_low);
								weight[3] = (fd - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fa > fd) {
								weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fd); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fd - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_high); 
								weight[3] = (fa - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else {
								weight[0] = (INTERVAL - fd); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fd - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high);
								weight[2] = (fc - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_high);
								weight[3] = (fa - fb); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
						}
					}
					else {
						if (fa > fc) {
							if (fc > fd) {
								weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fb - fa); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
								weight[2] = (fa - fc); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_low);
								weight[3] = (fc - fd); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low);
								weight[4] = fd;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fa > fd) {
                                weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fa); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fa - fd); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_low);
                                weight[3] = (fd - fc); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fb > fd) {
                                weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fd); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fd - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fa - fc); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
							else {
                                weight[0] = (INTERVAL - fd); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fb); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fb - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fa - fc); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
						}
						else if (fb > fc) {
							if (fa > fd) {
                                weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fc - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fa - fd); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low);
                                weight[4] = fd;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
                            else if (fc > fd){
                                weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fc - fd); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fd - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
                            else if (fb > fd){
                                weight[0] = (INTERVAL - fb); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fd); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fd - fc); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fc - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
                            else{
                                weight[0] = (INTERVAL - fd); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fb); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fb - fc); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fc - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
						}
						else {
							if (fa > fd) {
                                weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fb); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fb - fa); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fa - fd); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_low);
                                weight[4] = fd;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
							}
                            else if (fb > fd) {
                                weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fb); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fb - fd); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fd - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
                            else if (fc > fd) {
                                weight[0] = (INTERVAL - fc); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fd); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fd - fb); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_high);
                                weight[3] = (fb - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
                            else {
                                weight[0] = (INTERVAL - fd); points[0] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fc); 		points[1] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fc - fb); 		points[2] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_low, C_high, D_high);
                                weight[3] = (fb - fa); 		points[3] = acquire<INTERVAL + 1, SCALE>(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire<INTERVAL + 1, SCALE>(LUT, A_high, B_high, C_high, D_high);
                            }
						}
					}
					// 得到权重和指针了, 准备对 value00 - value11 赋值
					float temp00{0.f}, temp01{0.f}, temp10{0.f}, temp11{0.f};
					// 循环合并 0.06s → 0.049s
					for (int k = 0; k < CANDIDATE; ++k) {
						temp00 += weight[k] * points[k][0];
						temp01 += weight[k] * points[k][1];
						temp10 += weight[k] * points[k][2];
						temp11 += weight[k] * points[k][3];
					}
					// 赋值需要旋转
					switch (t) {
					case 0:
						value00 += temp00, value01 += temp01, value10 += temp10, value11 += temp11;
						break;
					case 1:
						value00 += temp10, value01 += temp00, value10 += temp11, value11 += temp01;
						break;
					case 2:
						value00 += temp11, value01 += temp10, value10 += temp01, value11 += temp00;
						break;
					case 3:
						value00 += temp01, value01 += temp11, value10 += temp00, value11 += temp10;
						break;
					default: 
						break;
					}
				}
				// 这里要赋 4 个值, 不连续的
				const int pos = 2 * (i * new_width + j);
				result[pos]                 = clip(nearest_round<unsigned char>(value00 / INTERVAL, 0.5f), 0, 255);
				result[pos + 1]             = clip(nearest_round<unsigned char>(value01 / INTERVAL, 0.5f), 0, 255);
				result[pos + new_width]     = clip(nearest_round<unsigned char>(value10 / INTERVAL, 0.5f), 0, 255);
				result[pos + new_width + 1] = clip(nearest_round<unsigned char>(value11 / INTERVAL, 0.5f), 0, 255);
			}
		}
		// 指针移动到下一个通道
		result       = result       + 2 * height * new_width;
		image_padded = image_padded + height_2 * width_2;
	}
}


extern "C" {
	void fast_lut_interpolation(
		unsigned char* result,
		unsigned char* image_padded,
		float* LUT,
		int height,
		int width, 
		int channel,
		int radius) {
		fast_lut_interpolation_inplementation<16, 4>(result, image_padded, LUT, height, width, channel, radius);
	}
}