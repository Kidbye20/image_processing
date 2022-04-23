// C++
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <functional>
// Eigen
#include <Eigen/SVD>
#include <Eigen/Dense>
// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>




namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
        std::cout << "是否连续" << std::boolalpha << one_image.isContinuous() << std::endl;
    }


    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type, const double times=2) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]) * times);
        return result;
    }
}





using crane_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;



int main() {
    const std::string image_path("./images/input/a0372-WP_CRW_6207.png");
    cv::Mat origin_image = cv::imread(image_path);
    assert(not origin_image.empty() and "图像读取失败");
    cv::cvtColor(origin_image, origin_image, cv::COLOR_BGR2GRAY);
    origin_image.convertTo(origin_image, CV_32F);
    const int H = origin_image.rows;
    const int W = origin_image.cols;
    crane_type A;
    cv::cv2eigen(origin_image, A);
    std::cout << A.rows() << ", " << A.cols() << ", " << std::endl;
    // 计算 SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_solver(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf U = svd_solver.matrixU();
    Eigen::MatrixXf V = svd_solver.matrixV();
    Eigen::MatrixXf S = svd_solver.singularValues();
    std::cout << U.rows() << ", " << U.cols() << ", " << U.size() << std::endl;
    std::cout << V.rows() << ", " << V.cols() << ", " << V.size() << std::endl;
    std::cout << S.rows() << ", " << S.cols() << ", " << S.size() << std::endl;
    // waiting
    return 0;
}
