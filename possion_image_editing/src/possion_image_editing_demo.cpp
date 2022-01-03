#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <bitset>



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

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    template<typename T>
    cv::Mat touint8(const std::vector<T>& source, const int H, const int W) {
        cv::Mat res(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) res.data[i] = std::abs(source[i]);
        return res;
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
}


std::vector<float> get_divergence(const cv::Mat& fore, const cv::Mat& back, const bool mix=true) {
    // 获取图像信息
    const int H = back.rows;
    const int W = back.cols;
    const int C = back.channels();
    const int length = H * W;
    // 计算
    std::vector<float> laplace_result(length * C, 0);
    const auto fore_padded = make_pad(fore, 1, 1);
    const auto back_padded = make_pad(back, 1, 1);
    const int W2 = back_padded.cols;
    std::vector<int> offset({-C, C, -W2 * C, W2 * C});
    for(int i = 0;i < H; ++i) {
        float* const res_ptr = laplace_result.data() + i * W * C;
        uchar* const fore_ptr = fore_padded.data + (1 + i) * W2 * C + C;
        uchar* const back_ptr = back_padded.data + (1 + i) * W2 * C + C;
        for(int j = 0;j < W * C; j += C) {
            // 三个方向上的, 还得分开计算
            for(int ch = 0;ch < C; ++ch) {
                float grad_sum = 0;
                // 四个方向上
                const int start = j + ch;
                for(int k = 0;k < 4; ++k) {
                    const float lhs = fore_ptr[start + offset[k]] - fore_ptr[start];
                    const float rhs = back_ptr[start + offset[k]] - back_ptr[start];
                    if(std::abs(lhs) > std::abs(rhs)) grad_sum += lhs;
                    else grad_sum += rhs;
                }
                res_ptr[j + ch] = grad_sum;
            }
        }
    }
    return laplace_result;
}

using cloned_type = std::vector< std::pair<int, std::vector<uchar> > >;
cloned_type build_and_solve_poisson_equations(
        const cv::Mat& back,
        const std::vector<float> divergence,
        const cv::Mat& mask) {
    // 获取图像信息
    const int H = back.rows;
    const int W = back.cols;
    const int C = back.channels();
    const int length = H * W;
    // 获取不规则区域内部(mask)的序号, 行优先; 非不规则区域为 -1, 不规则区域内部从 0 开始计数, 对应后面的 A, b 方程的行坐标
    std::vector<int> book(length, -1);
    int pixel_cnt = 0;
    for(int i = 0;i < length; ++i)
        if(mask.data[i] > 128)
            book[i] = pixel_cnt++;
    // 构建 Ax = b 线性方程, 需要填充 A 和 b 的值
    const int CH = back.channels();
    std::vector< Eigen::Triplet<float> > A_list; // 这个比 Eigen 稀疏矩阵 insert 要快很多
    A_list.reserve(pixel_cnt * 5);
    Eigen::MatrixXf b(pixel_cnt, CH);
    b.setZero();
    const uchar* const back_data = back.ptr<uchar>();
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            // 获取当前点, 判断在不在不规则区域范围内
            const int center = y * W + x;
            const int pid = book[center];
            if (pid == -1) continue;
            // A 的赋值
            A_list.emplace_back(pid, pid, -4.0);
            std::vector<float> missing(CH);
            std::vector<int> offset({-1, 1, -W, W});
            for(int ori = 0;ori < 4; ++ori) { // 上下左右四个点的赋值
                const int pos = center + offset[ori];
                if(book[pos] == -1) // 如果旁边这个点在边界上, b 的这一项要等于 0, 右边系数 -1
                    for(int k = 0;k < CH; ++k)
                        missing[k] += 1.f * back_data[CH * pos + k];
                else A_list.emplace_back(pid, book[pos], 1.0);
            }
            // b 的赋值
            for(int k = 0; k < CH; ++k)
                b(pid, k) = divergence[CH * center + k] - missing[k];
        }
    }
    // Eigen3 解稀疏矩阵的非线性方程组 Ax = b
    Eigen::SparseMatrix<float> A(pixel_cnt, pixel_cnt);
    A.setFromTriplets(A_list.begin(), A_list.end());
    A.makeCompressed();
    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
    solver.compute(A);
    // 道求解
    Eigen::MatrixXf X = solver.solve(b);
    // 把 Eigen3 的结果拷贝到背景图上
    cloned_type modified;
    modified.reserve(pixel_cnt);
    for(int i = 0;i < length; ++i) {
        if(book[i] != -1) {
            std::vector<uchar> temp(C);
            for(int ch = 0; ch < C; ++ch) temp[ch] = cv::saturate_cast<uchar>(X(book[i], ch));
            modified.emplace_back(i, temp);
        }
    }
    return modified;
}




cv::Mat possion_seamless_clone(
        const cv::Mat& foreground,
        const cv::Mat& background,
        const cv::Mat& mask,
        const std::pair<int, int> start) {
    // 异常处理
    assert(not foreground.empty() and "前景图 foreground 读取失败");
    assert(not background.empty() and "背景图 background 读取失败");
    assert(not mask.empty() and "mask 读取失败");
    assert(foreground.channels() == background.channels() and "前景图和背景图的通道数目不对等 !");
    assert(foreground.rows == mask.rows and foreground.cols == mask.cols and "前景图和 mask 的尺寸不对等 !");
    assert(start.first >= 0 and start.second >= 0 and "插入的起始位置不能为负 !");
    assert(start.first + foreground.rows <= background.rows and start.second + foreground.cols <= background.cols and "插入位置超出了背景图的界限 !");
    // 获取图像信息
    const int H = foreground.rows;
    const int W = foreground.cols;
    const int C = foreground.channels();

    // 背景区域对应位置切出来
    const auto background_crop = background(cv::Rect(start.second, start.first, W, H)).clone();

    // 求解要插入的内容的散度, 也可与背景图散度相融合
    const auto divergence = get_divergence(foreground, background_crop);

    // 根据泊松方程的条件, Ax = b, 构建 A, b 求解 x(不规则区域要填充的值)
    const auto modified = build_and_solve_poisson_equations(background_crop, divergence, mask);

    // 把结果到目标图像对应位置上修改像素值
    cv::Mat destination = background.clone();
    for(const auto & item : modified) {
        const int pos = (start.first + item.first / W) * destination.cols * C + (start.second + item.first % W) * C;
        for(int ch = 0; ch < C; ++ch) destination.data[pos + ch] = item.second[ch];
    }
    return destination;
}



void seamless_cloning_demo() {
    // 读取图像
    const std::string input_dir("../images/edit/3/");
    const std::string save_dir("./images/3/");
    cv::Mat background = cv::imread(input_dir + "background.jpg");
    cv::Mat foreground = cv::imread(input_dir + "src_image.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);

    cv::Mat result = possion_seamless_clone(foreground, background,mask, {134, 140});
    cv_show(result, "mixed and splited laplace");
    cv_write(result, save_dir + "mixed_splited_laplace.png");
}



int main() {

    cv::Mat background = cv::imread("../images/edit/3/background.jpg");
    cv::Mat foreground = cv::imread("../images/edit/3/src_image.png");
    cv::Mat mask = cv::imread("../images/edit/3/mask.png", cv::IMREAD_GRAYSCALE);

    cv::Mat result = possion_seamless_clone(foreground, background,mask, {134, 140});
    cv::imshow("Mixed Gradients", result);

    cv::imwrite("./images/output/3/mixed-gradients.png", result);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}




