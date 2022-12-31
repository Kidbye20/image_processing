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


class Timer:
    def __init__(self, message=''):
        self.message = message
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, trace):
        print(self.message + ' : {} s'.format(time.time() - self.start))



def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





# 读取训练好的 LUT, 注意 LUT 的数据类型 int8
scale       = 2
LUT_path    = "./Model_S_x{}_4bit_int8.npy".format(scale)
trained_LUT = numpy.load(LUT_path).reshape(-1, scale * scale).astype("float32")
print(trained_LUT.shape, trained_LUT.dtype)

# 读取待测试图像
image_path = "./images/a0015-DSC_0081.png"
low_res    = cv2.imread(image_path)
low_res    = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
height, width, channel = low_res.shape

# 对图像做 pad
radius = 1
low_res_padded = numpy.pad(low_res, [(radius, radius), (radius, radius), (0, 0)], mode="reflect")

# 三通道分开做, 做转置
low_res_padded = numpy.transpose(low_res_padded, (2, 0, 1))
low_res_padded = numpy.ascontiguousarray(low_res_padded)

# 把输入数据都放到 GPU 上
low_res_padded = cupy.asarray(low_res_padded)
trained_LUT    = cupy.asarray(trained_LUT)

# 准备一个输出在 GPU 上(注意内存排布是 channel, height, width)
result = cupy.zeros((channel, height * scale, width * scale), dtype="uint8")



# 准备 cuda kernel 代码
cuda_source = r'''
extern "C" {

	__device__ __forceinline__ float* acquire(float* LUT, int a, int b, int c, int d) {
		return LUT + 4 * (a * 17 * 17 * 17 + b * 17 * 17 + c * 17 + d);
	}

	__device__ __forceinline__ unsigned char round_and_clip(float x, float eps, unsigned char low, unsigned char high) {
		int data = int(x + eps);
		if (data < low) return low;
		else if (data > high) return high;
		return data;
	}

	__global__ void fast_lut_interpolation(
		unsigned char* result,
		unsigned char* source,
		float* LUT,
		int height,
		int width,
		int channel,
		int radius
	) {
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		int j = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < height && j < width) {
			// 算一些辅助变量
			const int height_2 = height + 2 * radius;
			const int width_2  = width  + 2 * radius;
			const int new_width = 2 * width;
			// 三通道分开做
			for (int c = 0; c < channel; ++c) {
				// 获取第 i 行有效数据的起始指针
				unsigned char* image_ptr = source + (radius + i) * width_2 + radius;
				// 集成结果, 应该会有 4 个值
				float value00 = 0.f, value01 = 0.f, value10 = 0.f, value11 = 0.f;
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
					// 得到当前的索引 [A][B][C][D], 得先除以 16(更改顺序并没有优化)
					int A_low  = A / 16;
					int B_low  = B / 16;
					int C_low  = C / 16;
					int D_low  = D / 16;
					int A_high = A_low + 1;
					int B_high = B_low + 1;
					int C_high = C_low + 1;
					int D_high = D_low + 1;
					// 得到 LUT 的 16 个顶点值(这部分去掉, 0.049s →  0.043s)
					// 求坐标的余数, 从而决定选哪 5 个顶点
					int fa = A % 16;
					int fb = B % 16;
					int fc = C % 16;
					int fd = D % 16;
					// 候选加权点
					float  weight[5];
					float* points[5];
					// 开始选点
					if (fa > fb) {
						if (fb > fc) {
							if (fc > fd) {
								weight[0] = 16 - fa;  points[0] = acquire(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fb;        points[1] = acquire(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fb - fc; 		points[2] = acquire(LUT, A_high, B_high, C_low, D_low); 
								weight[3] = fc - fd; 		points[3] = acquire(LUT, A_high, B_high, C_high, D_low); 
								weight[4] = fd;      		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
								
							} 
							else if (fb > fd) {
								weight[0] = 16 - fa; points[0] = acquire(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fb; 		points[1] = acquire(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fb - fd; 		points[2] = acquire(LUT, A_high, B_high, C_low, D_low); 
								weight[3] = fd - fc; 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
								
							}
							else if (fa > fd) {
								weight[0] = 16 - fa; points[0] = acquire(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fa - fd; 		points[1] = acquire(LUT, A_high, B_low, C_low, D_low); 
								weight[2] = fd - fb; 		points[2] = acquire(LUT, A_high, B_low, C_low, D_high); 
								weight[3] = fb - fc; 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
								
							}
							else {
								weight[0] = 16 - fd; points[0] = acquire(LUT, A_low, B_low, C_low, D_low); 
								weight[1] = fd - fa; 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high); 
								weight[2] = fa - fb; 		points[2] = acquire(LUT, A_high, B_low, C_low, D_high); 
								weight[3] = fb - fc; 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high); 
								weight[4] = fc;      		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
								
							}
						}
						else if (fa > fc) {
							if (fb > fd) {
					            weight[0] = (16 - fa); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
					            weight[1] = (fa - fc); 		points[1] = acquire(LUT, A_high, B_low, C_low, D_low);
					            weight[2] = (fc - fb); 		points[2] = acquire(LUT, A_high, B_low, C_high, D_low);
					            weight[3] = (fb - fd); 		points[3] = acquire(LUT, A_high, B_high, C_high, D_low);
					            weight[4] = fd ;       		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fc > fd) {
								weight[0] = (16 - fa); points[0] = acquire(LUT, A_low, B_low, C_low, D_low); 
		                        weight[1] = (fa - fc); 		points[1] = acquire(LUT, A_high, B_low, C_low, D_low);
		                        weight[2] = (fc - fd); 		points[2] = acquire(LUT, A_high, B_low, C_high, D_low);
		                        weight[3] = (fd - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
		                        weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							} 
							else if (fa > fd) {
								weight[0] = (16 - fa); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fa - fd); 		points[1] = acquire(LUT, A_high, B_low, C_low, D_low);
								weight[2] = (fd - fc); 		points[2] = acquire(LUT, A_high, B_low, C_low, D_high);
								weight[3] = (fc - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else {
								weight[0] = (16 - fd); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fd - fa); 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high);
								weight[2] = (fa - fc); 		points[2] = acquire(LUT, A_high, B_low, C_low, D_high);
								weight[3] = (fc - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
						}
						else {
							if (fb > fd) {
								weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fa); points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fa - fb); points[2] = acquire(LUT, A_high, B_low, C_high, D_low);
								weight[3] = (fb - fd); points[3] = acquire(LUT, A_high, B_high, C_high, D_low);
								weight[4] = fd;        points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fc > fd) {
								weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fa); 		points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fa - fd); 		points[2] = acquire(LUT, A_high, B_low, C_high, D_low);
								weight[3] = (fd - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fa > fd) {
								weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fc - fd); 		points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
								weight[2] = (fd - fa); 		points[2] = acquire(LUT, A_low, B_low, C_high, D_high); 
								weight[3] = (fa - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else {
								weight[0] = (16 - fd); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fd - fc); 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high);
								weight[2] = (fc - fa); 		points[2] = acquire(LUT, A_low, B_low, C_high, D_high);
								weight[3] = (fa - fb); 		points[3] = acquire(LUT, A_high, B_low, C_high, D_high);
								weight[4] = fb;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
						}
					}
					else {
						if (fa > fc) {
							if (fc > fd) {
								weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
								weight[1] = (fb - fa); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
								weight[2] = (fa - fc); 		points[2] = acquire(LUT, A_high, B_high, C_low, D_low);
								weight[3] = (fc - fd); 		points[3] = acquire(LUT, A_high, B_high, C_high, D_low);
								weight[4] = fd;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fa > fd) {
                                weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fa); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fa - fd); 		points[2] = acquire(LUT, A_high, B_high, C_low, D_low);
                                weight[3] = (fd - fc); 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else if (fb > fd) {
                                weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fd); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fd - fa); 		points[2] = acquire(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fa - fc); 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
							else {
                                weight[0] = (16 - fd); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fb); 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fb - fa); 		points[2] = acquire(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fa - fc); 		points[3] = acquire(LUT, A_high, B_high, C_low, D_high);
                                weight[4] = fc;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
						}
						else if (fb > fc) {
							if (fa > fd) {
                                weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fc); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fc - fa); 		points[2] = acquire(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fa - fd); 		points[3] = acquire(LUT, A_high, B_high, C_high, D_low);
                                weight[4] = fd;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
                            else if (fc > fd){
                                weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fc); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fc - fd); 		points[2] = acquire(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fd - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
                            else if (fb > fd){
                                weight[0] = (16 - fb); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fb - fd); 		points[1] = acquire(LUT, A_low, B_high, C_low, D_low);
                                weight[2] = (fd - fc); 		points[2] = acquire(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fc - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
                            else{
                                weight[0] = (16 - fd); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fb); 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fb - fc); 		points[2] = acquire(LUT, A_low, B_high, C_low, D_high);
                                weight[3] = (fc - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
						}
						else {
							if (fa > fd) {
                                weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fb); 		points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fb - fa); 		points[2] = acquire(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fa - fd); 		points[3] = acquire(LUT, A_high, B_high, C_high, D_low);
                                weight[4] = fd;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
							}
                            else if (fb > fd) {
                                weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fb); 		points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fb - fd); 		points[2] = acquire(LUT, A_low, B_high, C_high, D_low);
                                weight[3] = (fd - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
                            else if (fc > fd) {
                                weight[0] = (16 - fc); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fc - fd); 		points[1] = acquire(LUT, A_low, B_low, C_high, D_low);
                                weight[2] = (fd - fb); 		points[2] = acquire(LUT, A_low, B_low, C_high, D_high);
                                weight[3] = (fb - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
                            else {
                                weight[0] = (16 - fd); points[0] = acquire(LUT, A_low, B_low, C_low, D_low);
                                weight[1] = (fd - fc); 		points[1] = acquire(LUT, A_low, B_low, C_low, D_high);
                                weight[2] = (fc - fb); 		points[2] = acquire(LUT, A_low, B_low, C_high, D_high);
                                weight[3] = (fb - fa); 		points[3] = acquire(LUT, A_low, B_high, C_high, D_high);
                                weight[4] = fa;        		points[4] = acquire(LUT, A_high, B_high, C_high, D_high);
                            }
						}
					}
					// 得到权重和指针了, 准备对 value00 - value11 赋值
					float temp00 = 0.f, temp01 = 0.f, temp10 = 0.f, temp11 = 0.f;
					// 循环合并 0.06s → 0.049s
					for (int k = 0; k < 5; ++k) {
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
				result[pos]                 = round_and_clip(value00 / 16, 0.5f, 0, 255);
				result[pos + 1]             = round_and_clip(value01 / 16, 0.5f, 0, 255);
				result[pos + new_width]     = round_and_clip(value10 / 16, 0.5f, 0, 255);
				result[pos + new_width + 1] = round_and_clip(value11 / 16, 0.5f, 0, 255);

				// 移动到下一个通道
				result += 2 * height * new_width;
				source += height_2 * width_2;
			}
		}
	}
}
'''

cuda_module  = cupy.RawModule(code=cuda_source)
srlut_kernel = cuda_module.get_function("fast_lut_interpolation")
CUDA_CEIL = lambda x, y: int((x + y - 1) / y)
srlut_kernel(
	# grid
	(CUDA_CEIL(width, 32), CUDA_CEIL(height, 32)), 
	# block
	(32, 32),
	# arguements
	(result, low_res_padded, trained_LUT, height, width, channel, radius)
)

output = numpy.transpose(cupy.asnumpy(result), [1, 2, 0])[:, :, ::-1]
cv2.imwrite(image_path.replace(".png", "_SRresult.png"), output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv_show(output)