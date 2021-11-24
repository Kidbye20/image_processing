// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "guided_filter.h"


namespace {

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat box_filter(const cv::Mat& source, const int radius_h, const int radius_w) {
        // 首先我这里接收到的图片是 uchar, 但 type 是 1. / 255 的
        cv::Mat new_source;
        source.convertTo(new_source, CV_8UC1, 255);
        const int H = source.rows;
        const int W = source.cols;
        const int kernel_w = 2 * radius_w + 1;
        const int kernel_h = 2 * radius_h + 1;
        const int H2 = H - kernel_h;
        const int W2 = W - kernel_w;
        std::vector<double> sum(W2 * H2, 0.0);
        std::vector<double> buffer(W, 0.0);

        // 先累加前 kernel_h 行的 box
        for(int i = 0; i< kernel_h; ++i){
            uchar* const row_ptr = new_source.data + i * new_source.step;
            for(int j = 0; j < W; ++j) buffer[j] += row_ptr[j];
        }
        // 开始向右边和向下挪动, 计算每个点为中心的窗口累加值
        for(int i = 0; i < H2; ++i){
            // 当前 box 的累加值
            int cur_sum = 0;
            // 当前行第一个 box 的和
            for(int j = 0; j < kernel_w; ++j) cur_sum += buffer[j];
            const int _beg = i * W2;
            sum[_beg] = cur_sum;
            // 开始算这一行其它 box 的和, 减去左边去掉的一个, 加上右边加上的一个
            for(int j = 1; j < W2; ++j){
                cur_sum = cur_sum - buffer[j - 1] + buffer[j - 1 + kernel_w];
                sum[_beg + j] = cur_sum;
            }
            // 这一行求完了, 更新 buffer, 因为要向下挪
            uchar* const up_ptr = new_source.data + i * W;
            uchar* const down_ptr = new_source.data + (i + kernel_h) * W;
            for(int j = 0; j < W; ++j)
                buffer[j] = buffer[j] - up_ptr[j] + down_ptr[j];
        }
        //遍历，得到每个点的和，传给矩阵result
        cv::Mat result = new_source.clone();
        const int area = kernel_h * kernel_w;
        for(int i = radius_h + 1; i < H2; ++i){
            const int _beg = (i - radius_h) * W2;
            uchar* const row_ptr = result.data + i * H;
            for(int j = radius_w + 1; j < W - radius_w; ++j){
                const int pos = _beg + j - radius_w;
                row_ptr[j] = cv::saturate_cast<uchar>(sum[pos] / area);
            }
        }
        cv::Mat last_result;
        result.convertTo(last_result, CV_64FC1, 1. / 255);
        return last_result;
    }

    cv::Mat get_image_window_mean(const cv::Mat& source, const int radius) {
        cv::Mat mean_result = source.clone();
//        cv::boxFilter(source, mean_result, CV_64FC1, cv::Size(radius, radius));
        mean_result = box_filter(source, radius, radius);
        return mean_result;
    }

}


cv::Mat guided_filter(const cv::Mat& noise_image, const cv::Mat& guided_image, const int radius, const double eta) {
//    ------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
    cv::Mat srcMat, guidedMat;
	noise_image.convertTo(srcMat, CV_64FC1, 1.0 / 255);
	guided_image.convertTo(guidedMat, CV_64FC1, 1.0 / 255);
	//--------------【1】各种均值计算----------------------------------
	const auto mean_p = get_image_window_mean(srcMat, radius);
    cv_show(mean_p);
	std::cout << mean_p.rows << "---" << mean_p.cols << std::endl;
	// 39 254 161 18 224 63 206 73 94 12
	const auto mean_I = get_image_window_mean(guidedMat, radius);
	const auto mean_Ip = get_image_window_mean(srcMat.mul(guidedMat), radius);
	const auto mean_II = get_image_window_mean(guidedMat.mul(guidedMat), radius);
	//--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
	cv::Mat cov_Ip = mean_II.clone();
	const int length = noise_image.rows * noise_image.cols;
	std::cout << "length  " << length << std::endl;
//	for(int i = 0;i < length; ++i) {
//        cov_Ip.data[i] = mean_Ip.data[i] - mean_I.data[i] * mean_p.data[i];
//	}
	cov_Ip = mean_Ip - mean_I.mul(mean_p);
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);
	//---------------【3】计算参数系数a、b-------------------
	cv::Mat a = cov_Ip / (var_I + eta);
	cv::Mat b = mean_p - a.mul(mean_I);
	//--------------【4】计算系数a、b的均值-----------------
	const auto mean_a = get_image_window_mean(a, radius);
	const auto mean_b = get_image_window_mean(b, radius);
	//---------------【5】生成输出矩阵------------------
	cv::Mat dstImage = mean_a.mul(srcMat) + mean_b;
	cv::Mat result;
	dstImage.convertTo(result, CV_8UC1, 255);
	return result;
}
