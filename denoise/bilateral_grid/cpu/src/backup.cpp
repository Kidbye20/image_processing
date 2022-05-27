#include "utils.h"

namespace {
    template<typename T>
    inline T square(const T x) {
        return x * x;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

    template<typename T>
    T _fast_exp(T x) {
        x = 1.0 + x / 256;
        for(int i = 0;i < 8; ++i) x *= x;
        return x;
    }
}



cv::Mat bilateral_mean_filtering(
        cv::Mat noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=30) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 这里需要对图像做 paddding
    cv::copyMakeBorder(noisy_image, noisy_image, spatial_padding, spatial_padding, spatial_padding, spatial_padding, cv::BORDER_REFLECT);
    const int W2 = noisy_image.cols;
    // 准备一个偏移量模板
    int max_k = 0;
    std::vector<int> offset(square<int>(2 * spatial_padding + 1));
    for(int i = -spatial_padding; i <= spatial_padding; ++i)
        for(int j = -spatial_padding; j <= spatial_padding; ++j)
            offset[max_k++] = i * W2 + j;
    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, noisy_image.type());
    uchar* const res_ptr = result.ptr<uchar>();
    // 开始卷积
    for(int i = 0; i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + (spatial_padding + i) * W2 + spatial_padding;
        for(int j = 0; j < W; ++j) {
            // 获取这个点的亮度
            int intensity = row_ptr[j];
            // 做低通滤波
            float weight_sum = 0.f;
            float intensity_sum = 0.f;
            for(int k = 0; k < max_k; ++k) {
                const int pos = offset[k];
                // 如果邻域点和当前点的亮度差小于一定程度, 才参与加权
                if(std::abs(row_ptr[j + pos] - intensity) <= intensity_padding) {
                    weight_sum += 1;
                    intensity_sum += row_ptr[j + pos];
                }
                // 注释上面三行, 放下面两行代码, 做朴素的均值滤波
                // weight_sum += 1;
                // intensity_sum += row_ptr[j + pos];
            }
            res_ptr[cnt++] = cv::saturate_cast<uchar>(intensity_sum / weight_sum);
        }
    }
    return result;
}


cv::Mat bilateral_grid_mean_filtering(
        cv::Mat& noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=30,
        const int intensity_level=255) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 构造一个网格
    const int grid_height = H + 2 * spatial_padding;
    const int grid_width = W + 2 * spatial_padding;
    const int grid_intensity = intensity_level + 2 * intensity_padding;
    const int grid_size = grid_height * grid_width * grid_intensity;
    std::vector<float> grid(grid_size, 0);
    // 把图像信息填充到网格中
    for(int i = 0;i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + i * W;
        for(int j = 0;j < W; ++j) {
            int intensity = row_ptr[j];
            grid[((i + spatial_padding) * grid_width + j + spatial_padding) * grid_intensity + intensity + intensity_padding] += intensity;
        }
    }
    // 构造一个偏移量模板
    int max_k = 0;
    std::vector<int> offset((2 * spatial_padding + 1) * (2 * spatial_padding + 1) * (2 * intensity_padding + 1));
    for(int i = -spatial_padding;i <= spatial_padding; ++i)
        for(int j = -spatial_padding;j <= spatial_padding; ++j)
            for(int k = -intensity_padding;k <= intensity_padding; ++k)
                offset[max_k++] = (i * grid_width + j) * grid_intensity + k;
    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, noisy_image.type());
    uchar* const res_ptr = result.ptr<uchar>();
    // 开始卷积
    for(int i = spatial_padding, max_i = grid_height - spatial_padding; i < max_i; ++i) {
        for(int j = spatial_padding, max_j = grid_width - spatial_padding; j < max_j; ++j) {
            // 获取这个位置的亮度值
            int intensity = noisy_image.data[(i - spatial_padding) * W + j - spatial_padding];
            // 定位网格
            float* const grid_start = grid.data() + (i * grid_width + j) * grid_intensity + intensity_padding;
            // 以 grid_start[intensity] 为中心, 做以此低通滤波
            float weight_sum = 0.f;
            float intensity_sum = 0.f;
            for(int k = 0;k < max_k; ++k) {
                const int pos = offset[k];
                if(grid_start[intensity + pos] > 0) {
                    weight_sum += 1;
                    intensity_sum += grid_start[intensity + pos];
                }
            }
            // 卷积输出填充到对应的位置上
            res_ptr[cnt++] = cv::saturate_cast<uchar>(intensity_sum / weight_sum);
        }
    }
    return result;
}




cv::Mat bilateral_grid_mean_filtering_faster(
        cv::Mat noisy_image,
        const float spatial_sigma=1.0,
        const int intensity_padding=7,
        const int intensity_level=64) {
    assert(noisy_image.channels() == 1 and noisy_image.type() == CV_8UC1 and "only gray images are supported !");
    // 获取图像信息
    const int H = noisy_image.rows;
    const int W = noisy_image.cols;
    // 如果是其他数据类型, 还得转成 float
    // 获取滤波核参数
    const int spatial_padding = std::ceil(3 * spatial_sigma);
    // 构造一个网格
    const int grid_height = H + 2 * spatial_padding;
    const int grid_width = W + 2 * spatial_padding;
    const int grid_intensity = intensity_level + 2 * intensity_padding;
    const int grid_size = grid_height * grid_width * grid_intensity;
    std::vector<float> grid(grid_size, 0);
    std::vector<float> grid_weight(grid_size, 0);
    // 每个网格的长度
    const int grid_interval = std::ceil(255 / intensity_level);
    // 把图像信息填充到网格中
    for(int i = 0;i < H; ++i) {
        uchar* const row_ptr = noisy_image.ptr<uchar>() + i * W;
        for(int j = 0;j < W; ++j) {
            int intensity = row_ptr[j];
            int pos = static_cast<int>(intensity * 1.f / float(grid_interval));
            pos = ((i + spatial_padding) * grid_width + j + spatial_padding) * grid_intensity + pos + intensity_padding;
            grid[pos] += intensity;
            grid_weight[pos] += 1;
        }
    }
    // 准备一个偏移量模板
    int max_k = 0;
    std::vector<int> offset(square<int>(2 * spatial_padding + 1) * (2 * intensity_padding + 1));
    for(int i = -spatial_padding; i <= spatial_padding; ++i)
        for(int j = -spatial_padding; j <= spatial_padding; ++j)
            for(int k = -intensity_padding; k <= intensity_padding; ++k)
                offset[max_k++] = (i * grid_width + j) * grid_intensity + k;
    // 开始在网格中卷积
    std::vector<float> grid_result(grid_size, 0);
    std::vector<float> grid_weight_result(grid_size, 0);

    for(int i = spatial_padding, max_i = grid_height - spatial_padding; i < max_i; ++i) {
        for(int j = spatial_padding, max_j = grid_width - spatial_padding; j < max_j; ++j) {
            // 获取这个点在网格中的位置
            for(int pos = intensity_padding, max_p = grid_intensity - intensity_padding; pos < max_p; ++pos) {
                // 获取在网格中的偏移量
                float* const grid_ptr = grid.data() + (i * grid_width + j) * grid_intensity + pos + intensity_padding;
                float* const weight_ptr = grid_weight.data() + (i * grid_width + j) * grid_intensity + pos + intensity_padding;
                // 开始卷积一个点
                float weight_sum = 0.f;
                float intensity_sum = 0.f;
                for(int k = 0; k < max_k; ++k) {
                    const int p = offset[k];
                    {
                        weight_sum += weight_ptr[p];
                        intensity_sum += grid_ptr[p];
                    }
                }
                // 卷积结束, 得到这个格子的值
//                std::cout << value << std::endl;
                // 根据结果来网格中求值
                grid_result[(i * grid_width + j) * grid_intensity + pos + intensity_padding] = intensity_sum / max_k;
                grid_weight_result[(i * grid_width + j) * grid_intensity + pos + intensity_padding] = weight_sum / max_k;
            }
        }
    }

    auto trilinear_interpolate = [](
            const std::vector<float>& wi_grid,
            const std::vector<float>& w_grid,
            const float x, const float y, const float z,
            std::vector<int> border) ->float {
        // 计算这个小数坐标 (x, y, z) 在网格中, 在三个方向上的上界和下界
        const int x_down = clip<int>(std::floor(x), 0, border[0] - 1);
        const int x_up   = clip<int>(x_down + 1, 0, border[0] - 1);
        const int y_down = clip<int>(std::floor(y), 0, border[1] - 1);
        const int y_up   = clip<int>(y_down + 1, 0, border[1] - 1);
        const int z_down = clip<int>(std::floor(z), 0, border[2] - 1);
        const int z_up   = clip<int>(z_down + 1, 0, border[2] - 1);
        // 获取这个小数坐标在 x, y, z 方向上的权重量
        const float x_weight = std::abs(x - x_down);
        const float y_weight = std::abs(y - y_down);
        const float z_weight = std::abs(z - z_down);
        // 计算 (__x, __y, __z) 在网格中的偏移地址
        auto index = [&](const int _x, const int _y, const int _z) ->int {
            return (_x * border[1] + _y) * border[2] + _z;
        };
        // 准备立方体 8 个点坐标对应的偏移量
        std::vector<int> offsets = {
            index(x_down, y_down, z_down),
            index(x_up,   y_down, z_down),
            index(x_down, y_up,   z_down),
            index(x_down, y_down, z_up),
            index(x_up,   y_up,   z_down),
            index(x_up,   y_down, z_up),
            index(x_down, y_up,   z_up),
            index(x_up,   y_up,   z_up)
        };
        // 准备立方体 8 个点坐标对应的加权值
        std::vector<float> weights = {
            (1.f - x_weight) * (1.f - y_weight) * (1.f - z_weight),
            x_weight         * (1.f - y_weight) * (1.f - z_weight),
            (1.f - x_weight) * y_weight         * (1.f - z_weight),
            (1.f - x_weight) * (1.f - y_weight) * z_weight,
            x_weight         * y_weight         * (1.f - z_weight),
            x_weight         * (1.f - y_weight) * z_weight,
            (1.f - x_weight) * y_weight         * z_weight,
            x_weight         * y_weight         * z_weight
        };
        // 两个网格的插值共用一套加权参数
        float wi_interpolated = 0.f;
        for(int i = 0;i < 8; ++i) wi_interpolated += weights[i] * wi_grid[offsets[i]];
        float w_interpolated = 0.f;
        for(int i = 0;i < 8; ++i) w_interpolated += weights[i] * w_grid[offsets[i]];
        // 插值结果相除, 归一化
        return wi_interpolated / w_interpolated;
    };

    // 准备一个结果
    int cnt = 0;
    cv::Mat result(H, W, CV_8UC1);
    uchar* const res_ptr = result.ptr<uchar>();
    // 对于结果中每一个点, 去 grid_result 中去插值得到结果
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 计算这个点在网格中的坐标
            const float x = i + spatial_padding;
            const float y = j + spatial_padding;
            const float z = noisy_image.data[i * W + j] * 1.f / float(grid_interval) + intensity_padding;
            // 三次线性插值, 两个分支
            float interp_res = trilinear_interpolate(grid_result, grid_weight_result, x, y, z, {grid_height, grid_width, grid_intensity});
            // wi / w 是最终的加权结果
            res_ptr[cnt++] = cv::saturate_cast<uchar>(interp_res);
        }
    }
    return result;
}










cv::Mat fast_bilateral_approximation(
        cv::Mat input,
        // 采样率
        const float range_sample=0.1,
        // 网格做滤波时候的半径
        const int grid_padding=2,
        // 打印信息
        const bool verbose=true) {

    using pointer_type = const float* const;

    input.convertTo(input, CV_32FC1);
    input /= 255;
    cv::Mat& refer = input;

    // 【1】********************** 收集图像信息 **********************
    const int H = input.rows;
    const int W = input.cols;
    const int length = H * W;
    assert(H == refer.rows and W == refer.cols and "shapes of input and refer must be the same");
    assert(input.channels() == 1 and refer.channels() == 1 and "only gray images are supported");

    // 【2】********************** 根据图像的宽高, 亮度构建双边网格 **********************
    // 计算图像中的取值范围, 用于定义值域网格
    pointer_type refer_ptr = refer.ptr<float>();
    const float range_min = min_in_array(refer_ptr, length);
    const float range_max = max_in_array(refer_ptr, length);
    const float range_interval = range_max - range_min;
    if(verbose) {
        std::cout << "intensity    :  [" << range_min << ", " << range_max << "]" << std::endl;
    }
    // 决定下采样网格的大小
    const int grid_height = H + 2 * grid_padding;
    const int grid_width = W + 2 * grid_padding;
    const int grid_value = std::floor(range_interval / range_sample) + 1 + 2 * grid_padding;
    // 创建 grid, 一个是分母的加权部分, 另一个是分子(齐次的 1)
    const int grid_size = grid_height * grid_width * grid_value;
    std::vector<float> wi_grid(grid_size, 0);
    std::vector<float> w_grid(grid_size, 0);
    if(verbose) {
        std::cout << "grid  :  \n";
        std::cout << "\theight     :  " << grid_height << std::endl;
        std::cout << "\twidth      :  " << grid_width << std::endl;
        std::cout << "\tvalue      :  " << grid_value << std::endl;
        std::cout << "\tgrid_size  :  "<< grid_size << std::endl;
    }
    // 根据参考图像的信息, 将输入图像下采样填充到 grid 网格中
    for(int i = 0;i < H; ++i) {
        const int x = i + grid_padding;  // 图像第 i 行映射到网格中的坐标
        pointer_type I_ptr = input.ptr<float>() + i * W;  // 输入图象在第 i 行的指针
        pointer_type R_ptr = refer_ptr + i * W;           // 参考图像在第 i 行的指针
        for(int j = 0;j < W; ++j) {
            const int y = j + grid_padding;  // 图像第 j 列映射到网格中的坐标
            const int z = std::floor((R_ptr[j] - range_min) * 1.f / range_sample) + grid_padding + 1;  // 图像中点 (i,j) 的亮度值映射到网格的 z 维的坐标
            const int grid_pos = (x * grid_width + y) * grid_value + z;
            wi_grid[grid_pos] += I_ptr[j];
            w_grid[grid_pos] += 1;
        }
    }
    if(verbose) {
        int effective_count = 0;
        for(int i = 0;i < grid_size; ++i)
            if(w_grid[i] > 0) ++effective_count;
        std::cout << "filling proportion of the grid is  " << effective_count * 1.f / grid_size << std::endl;
    }

    // ********************** 在网格上做卷积, 低通滤波 **********************
    std::vector<int> offset({grid_width * grid_value, grid_value, 1});
    std::vector<float> wi_grid_buffer(grid_size, 0); // 用于存储多维分离卷积的上一次结果
    std::vector<float> w_grid_buffer(grid_size, 0);
    for(int dimension = 0;dimension < 3; ++dimension) {
        const int _offset = offset[dimension];  // 当前维度 +1, -1 在网格中的偏移量
        for(int iter = 0;iter < 4; ++iter) {     // 实际半径为 2 倍的 1, 下面的滤波半径都是 1
            wi_grid.swap(wi_grid_buffer);
            w_grid.swap(w_grid_buffer);       // 这个交换很巧妙, 第一次卷积的结果存放在 buffer, 第二次从 buffer 中再卷积一次放在网格中
            // 开始三维卷积
            for(int i = 1, i_MAX = grid_height - 1; i < i_MAX; ++i) {
                for(int j = 1, j_MAX = grid_width - 1; j < j_MAX; ++j) {
                    const int start = (i * grid_width + j) * grid_value; // 当前网格在第(i, j)个格子的偏移量
                    float* wi = wi_grid.data() + start;
                    float* wi_buf = wi_grid_buffer.data() + start;       // 加权的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    float* w = w_grid.data() + start;
                    float* w_buf = w_grid_buffer.data() + start;         // 齐次的网格 和 它的上一次卷积结果, 在第(i, j)个格子的偏移地址
                    for(int k = 1, k_MAX = grid_value - 1; k < k_MAX; ++k) {
                        // 每次卷积, dimension 这个维度上前一个像素 + 后一个像素 和 当前像素做加权平均, 平滑
                        wi[k] = 0.25 * (2.0 * wi_buf[k] + wi_buf[k - _offset] + wi_buf[k + _offset]);
                        w[k] = 0.25 * (2.0 * w_buf[k] + w_buf[k - _offset] + w_buf[k + _offset]);
                    }
                }
            }
        }
    }
    if(verbose) {
        std::cout << "low-pass convolution on grid is completed" << std::endl;
    }

    // ********************** 网格做了低通滤波之后, 根据参考图从网格中插值得到每一个目标点的值 **********************
    auto trilinear_interpolate = [](
            const std::vector<float>& wi_grid,
            const std::vector<float>& w_grid,
            const float x, const float y, const float z,
            std::vector<int> border) ->float {
        // 计算这个小数坐标 (x, y, z) 在网格中, 在三个方向上的上界和下界
        const int x_down = clip<int>(std::floor(x), 0, border[0] - 1);
        const int x_up   = clip<int>(x_down + 1, 0, border[0] - 1);
        const int y_down = clip<int>(std::floor(y), 0, border[1] - 1);
        const int y_up   = clip<int>(y_down + 1, 0, border[1] - 1);
        const int z_down = clip<int>(std::floor(z), 0, border[2] - 1);
        const int z_up   = clip<int>(z_down + 1, 0, border[2] - 1);
        // 获取这个小数坐标在 x, y, z 方向上的权重量
        const float x_weight = std::abs(x - x_down);
        const float y_weight = std::abs(y - y_down);
        const float z_weight = std::abs(z - z_down);
        // 计算 (__x, __y, __z) 在网格中的偏移地址
        auto index = [&](const int _x, const int _y, const int _z) ->int {
            return (_x * border[1] + _y) * border[2] + _z;
        };
        // 准备立方体 8 个点坐标对应的偏移量
        std::vector<int> offsets = {
            index(x_down, y_down, z_down),
            index(x_up,   y_down, z_down),
            index(x_down, y_up,   z_down),
            index(x_down, y_down, z_up),
            index(x_up,   y_up,   z_down),
            index(x_up,   y_down, z_up),
            index(x_down, y_up,   z_up),
            index(x_up,   y_up,   z_up)
        };
        // 准备立方体 8 个点坐标对应的加权值
        std::vector<float> weights = {
            (1.f - x_weight) * (1.f - y_weight) * (1.f - z_weight),
            x_weight         * (1.f - y_weight) * (1.f - z_weight),
            (1.f - x_weight) * y_weight         * (1.f - z_weight),
            (1.f - x_weight) * (1.f - y_weight) * z_weight,
            x_weight         * y_weight         * (1.f - z_weight),
            x_weight         * (1.f - y_weight) * z_weight,
            (1.f - x_weight) * y_weight         * z_weight,
            x_weight         * y_weight         * z_weight
        };
        // 两个网格的插值共用一套加权参数
        float wi_interpolated = 0.f;
        for(int i = 0;i < 8; ++i) wi_interpolated += weights[i] * wi_grid[offsets[i]];
        float w_interpolated = 0.f;
        for(int i = 0;i < 8; ++i) w_interpolated += weights[i] * w_grid[offsets[i]];
        // 插值结果相除, 归一化
        return wi_interpolated / w_interpolated;
    };

    int cnt = 0;
    cv::Mat result(H, W, CV_8UC1);  // 准备一个结果
    uchar* const res_ptr = result.ptr<uchar>();
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            // 计算这个点在网格中的坐标
            const float x = i + grid_padding;
            const float y = j + grid_padding;
            const float z = (refer_ptr[i * W + j] - range_min) / range_sample + grid_padding;
            // 三次线性插值, 两个分支
            float interp_res = trilinear_interpolate(wi_grid, w_grid, x, y, z, {grid_height, grid_width, grid_value});
            // wi / w 是最终的加权结果
            res_ptr[cnt++] = cv::saturate_cast<uchar>(255 * interp_res);
        }
    }
    return result;
}






int main() {
    std::setbuf(stdout, 0);

    // 读取图像
    cv::Mat noisy_image = cv::imread("./images/input/example.png", 0);
    cv::resize(noisy_image, noisy_image, {50, 50});

    // 用最暴力的网格做均值滤波
//    auto smoothed = bilateral_grid_mean_filtering(
//            noisy_image, 1.5, 30, 255);

    // 优化上面, 对亮度域做分级, 加速
//    auto smoothed = bilateral_grid_mean_filtering_faster(
//

    auto smoothed = fast_bilateral_approximation(noisy_image, 0.05, 2, true);

    // 单纯用均值滤波
    // auto smoothed = bilateral_mean_filtering(noisy_image, 1.5, 30);



    cv_show(cv_concat({noisy_image, smoothed}));

    return 0;
}