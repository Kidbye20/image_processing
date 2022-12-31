# Python
import os
import sys
import time
# 3rd party
import cv2
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
interval    = 16
LUT_path    = "./Model_S_x{}_4bit_int8.npy".format(scale)
trained_LUT = numpy.load(LUT_path).reshape(-1, scale * scale).astype("float32")
print(trained_LUT.shape, trained_LUT.dtype)

# 读取待测试图像
image_path = "./images/a0015-DSC_0081.png"
low_res    = cv2.imread(image_path)
low_res    = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
height, width, channel = low_res.shape

# 对图像做 pad
low_res_padded = numpy.pad(low_res, [(1, 1), (1, 1), (0, 0)], mode="reflect")

# 三通道分开做, 做转置
low_res_padded = numpy.transpose(low_res_padded, (2, 0, 1))
low_res_padded = numpy.ascontiguousarray(low_res_padded)
print(low_res_padded.shape)

# 加载我自己的库
os.system("g++ -fPIC -shared -O2 ./fast_lut_interpolation.cpp -o ./fast_lut_interpolation.so")
lib = ctypes.cdll.LoadLibrary("./fast_lut_interpolation.so")

# 生成一个结果
result = numpy.zeros((channel, height * scale, width * scale), dtype="uint8")
print(result.shape, low_res_padded.shape, low_res_padded.shape[-1])
	
with Timer("推理 ") as infer_scope:
	lib.fast_lut_interpolation(
		result.ctypes.data_as(ctypes.c_char_p), 
		low_res_padded.ctypes.data_as(ctypes.c_char_p), 
		trained_LUT.ctypes.data_as(ctypes.c_char_p), 
		height, 
		width,
		channel,
		1
	)

print("result  ", result.shape)
result = numpy.transpose(result, (1, 2, 0))
# 转到 RGB
result = result[:, :, ::-1]
cv2.imwrite(image_path.replace(".png", "_SRresult.png"), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv_show(result)
