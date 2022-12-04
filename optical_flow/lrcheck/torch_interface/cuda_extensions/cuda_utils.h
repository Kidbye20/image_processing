// C++
#include <math.h>
#include <assert.h>
#include <cstdio>
#include <iostream>
// CUDA
#include <cuda_runtime.h>


inline int CUDA_CEIL(const int x, const int y) {
	return (x + y - 1) / y;
}

__forceinline__ __device__ float float_clip(const float x, const float low, const float high) {
	if (x < low) return low;
	else if (x > high) return high;
	else return x;
}

__forceinline__ __device__ unsigned char uint8_clip(const float x, const unsigned char low, const unsigned char high) {
	if (x < low) return low;
	else if (x > high) return high;
	else return unsigned char(x);
}