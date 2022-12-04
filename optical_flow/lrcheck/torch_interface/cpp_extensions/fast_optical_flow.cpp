// C++
#include <cmath>
#include <vector>
#include <iostream>
// torch
#include <torch/extension.h>





// ================================= CUDA 函数定义, 提供给 cpp =====================
void optical_flow_warp_cuda(
		unsigned char* result, 
		const unsigned char* image, 
		const float* flow, 
		const int batch_size, 
		const int channel,
		const int height, 
		const int width);


void optical_flow_lrcheck_cuda(
		unsigned char* occulusion, 
		const float* forward_flow, 
		const float* backward_flow, 
		const int batch_size, 
		const int channel,
		const int height, 
		const int width, 
		const float dis_threshold);






// ================================= torch 接口, 决定使用 CUDA/cpu =====================
void optical_flow_warp(
		at::Tensor& result, 
		const at::Tensor& image, 
		const at::Tensor& flow) {
	// 检查是否支持 cuda
	if (image.is_cuda()) {
#ifdef WITH_CUDA
		optical_flow_warp_cuda(
			result.data_ptr<unsigned char>(), 
			image.data_ptr<unsigned char>(), 
			flow.data_ptr<float>(), 
			image.size(0), image.size(1), image.size(2), image.size(3));
#else
		AT_ERROR("Function 'optical_flow_warp_cuda' is not complied with GPU support!");
#endif
	}
	else {
		AT_ERROR("Function 'optical_flow_warp_cpu' is not complied with GPU support!");
	}
}



// 我这种写法还是不太对, 我应该把接口都分开放置, 比如 cuda 跟 torch
void optical_flow_lrcheck(
		at::Tensor& occulusion, 
		const at::Tensor& forward_flow, 
		const at::Tensor& backward_flow,
		const float dis_threshold) {
	// 检查是否支持 cuda
	if (forward_flow.is_cuda()) {
#ifdef WITH_CUDA
		optical_flow_lrcheck_cuda(
			occulusion.data_ptr<unsigned char>(), 
			forward_flow.data_ptr<float>(), 
			backward_flow.data_ptr<float>(), 
			forward_flow.size(0), 
			forward_flow.size(1), 
			forward_flow.size(2), 
			forward_flow.size(3), 
			dis_threshold);
#else
		AT_ERROR("Function 'optical_flow_lrcheck_cuda' is not complied with GPU support!");
#endif
	}
	else {
		AT_ERROR("Function 'optical_flow_lrcheck' is not complied with GPU support!");
	}
}






// ================================= torch → Python 接口=====================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
	m.def(
		"warp", 
		&optical_flow_warp, 
		"warp flow with GPU/CPU support", 
		py::arg("result"), 
		py::arg("image"), 
		py::arg("flow")
	);
	m.def(
		"lrcheck", 
		&optical_flow_lrcheck, 
		"forward for optical flow check with GPU/CPU support", 
		py::arg("occulusion"), 
		py::arg("forward_flow"), 
		py::arg("backward_flows"),
		py::arg("dis_threshold")=1.f
	);
#endif
}