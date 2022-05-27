#include "utils.h"
#include <random>


template<typename T>
class MonotonousQueue {
private:
    const int capacity;   // 窗口容量
    std::list<T> Q;       // 单调队列
    T* data;              // 数据指针
    std::function<T(const T, const T)> compare;
public:
    MonotonousQueue(T* _data, const int _capacity, const std::function<T(const T, const T)> _comp)
        : data(_data), capacity(_capacity), compare(_comp) {}

    // 插入窗口最右边的元素, 检查是否保持单调
    void emplace(const int i) {
        // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
        while(not Q.empty()) {
            if(this->compare(data[i], data[Q.back()]))  // 如果队列中都比当前元素小, 停止
                Q.pop_back();
            else break; // 否则, 把队列中大于当前元素的 pop_back 掉
        }
        // 当前元素的坐标放到这里
        Q.emplace_back(i);
//        std::cout << "单调队列  ";
//        for(const auto it : Q) std::cout << it << "  ";
//        std::cout << "\n";
        // 如果当前维护的区间长度超出了窗口
//        std::cout << i << ", " << Q.front() << ", " << i - Q.front() << "\tcapacity = " << capacity << std::endl;
        if(i - Q.front() == capacity)
            Q.pop_front();
    }

    T front() const {
        return this->Q.front();
    }
};


void test_1d_extremum_filtering() {
    // 生成一个一维的数据
    std::vector<int> image_data({4, 1, 3, 0, 8, 9, -1, 2});
    // 设定滤波核的长度
    const int kernel_size = 3;
    // 最大值滤波 or 最小值滤波
    constexpr bool MIN_FILTER = true;
    // 决定 padding 的值
    constexpr int EXTRENUM = MIN_FILTER ? 1 << 20 : -(1 << 20);
    auto comp = MIN_FILTER ? [](const int lhs, const int rhs) {return lhs < rhs;} : [](const int lhs, const int rhs) {return lhs > rhs;} ;
    // 对数据做 padding
    for(int i = 0; i < kernel_size; ++i)
        image_data.emplace_back(EXTRENUM);
    // 获取数据信息
    const int image_size = image_data.size();
    int* image_ptr = image_data.data();
    // 准备一个滤波结果
    const int result_size = image_size - kernel_size;
    std::vector<int> result(result_size);
    // 维护一个单调队列
    MonotonousQueue<int> Q(image_ptr, kernel_size, comp);
    Q.emplace(0);
    // 开始最小值滤波
    for(int i = 1; i < image_size; ++i) {
        // 如果 i >= kernel, 说明当前窗口内已经记录了第 i - kernel 个窗口的最值
        if(i >= kernel_size)
            result[i - kernel_size] = image_data[Q.front()];
        // 尝试把第 i 个数据
        Q.emplace(i);
    }
    // 展示最值滤波结果
    for(int i = 0; i < result_size; ++i)
        std::cout << result[i] << "  ";
    std::cout << std::endl;
}



void test_2d_extremum_filtering() {
    // 定义一个二维的数据
    const int H = 4;
    const int W = 6;
    std::vector<int> image_data(H * W);
    // 定义随机种子
    std::default_random_engine seed(212);
    std::uniform_int_distribution engine(0, 20);
    for(int i = 0, L = H * W; i < L; ++i)
        image_data[i] = engine(seed);
    // 打印
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < W; ++j)
            std::cout << image_data[i * W + j] << "  ";
        std::cout << std::endl;
    }

    // 设定滤波的参数(默认是最小值滤波)
    const int kernel_size = 3;
    assert(kernel_size & 1);
    const int radius = (kernel_size - 1) >> 1;
    const int EXTREMUM = 1 << 20;
    auto comp = [](const int l, const int r){return l <= r;};
    // 对数据做 padding
    std::cout << "对数据做 padding\n";
    const int H2 = H + 2 * radius;
    const int W2 = W + 2 * radius;
    std::vector<int> padded_data(H2 * W2, EXTREMUM);
    for(int i = 0; i < H; ++i) {
        int* const src_ptr = image_data.data() + i * W;
        int* const des_ptr = padded_data.data() + (i + radius) * W2 + radius;
        for(int j = 0; j < W; ++j)
            des_ptr[j] = src_ptr[j];
    }
    //
    for(int i = 0; i < H2; ++i) {
        for(int j = 0; j < W2; ++j)
            std::cout << padded_data[i * W2 + j] << "  ";
        std::cout << std::endl;
    }

    // 下一步, 准备做最小值滤波, 先做 H 行的最小值滤波
    std::cout << "做水平方向上的最小值滤波\n";
    std::vector<int> temp;
    for(int i = 0; i < H; ++i) {
        // 对第 i 行做一次最小值滤波
        int* const row_ptr = padded_data.data() + i * W2;
        MonotonousQueue<int> Q(row_ptr, kernel_size, comp);  // 直接 clear, empty 置换, 或者 list 改成数组
        Q.emplace(0);
        for(int j = 1; j < W2; ++j) {
            if(j >= kernel_size)
                row_ptr[j - kernel_size + radius] = row_ptr[Q.front()];
            Q.emplace(j);
        }
    }
    for(int i = 0; i < H2; ++i) {
        for(int j = 0; j < W2; ++j)
            std::cout << padded_data[i * W2 + j] << "  ";
        std::cout << std::endl;
    }
    // 做 W 列的最小值滤波
    std::cout << "做竖直方向上的最小值滤波\n";
    for(int i = 0; i < W; ++i) {
        // 取出这一列的第一个元素的偏移量
        int* const col_ptr = padded_data.data() + i;
        // 这里的 capacity 很坑
        MonotonousQueue<int> Q(col_ptr, kernel_size * W2, comp);
        Q.emplace(i);
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size) {
                col_ptr[(j - kernel_size) * W2 + i] = col_ptr[Q.front()];
                if(i == 0) {
//                    std::cout << j - kernel_size << ", " << (j - kernel_size) * W2 + i << "====>  " << Q.front() << ", " << col_ptr[Q.front()] << "\n";
                }
            }
            Q.emplace(j * W2 + i);
        }
    }
    for(int i = 0; i < H2; ++i) {
        for(int j = 0; j < W2; ++j)
            std::cout << padded_data[i * W2 + j] << "  ";
        std::cout << std::endl;
    }
}


int main() {
    std::setbuf(stdout, 0);


//    // 测试一维的最小值滤波
//    test_1d_extremum_filtering();

    // 测试一维的最小值滤波
    test_2d_extremum_filtering();

    return 0;
}



