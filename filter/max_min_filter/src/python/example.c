#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <stdbool.h>

typedef unsigned char data_type;


// 小于(等于号不可以丢)
bool less_than(data_type l, data_type r) {
    return l <= r;
}

// 大于(等于号不可以丢)
bool greater_than(data_type l, data_type r) {
    return l >= r;
}


// 快速最小值滤波
data_type* fast_min_filtering(
        data_type* src,     // 待滤波图像
        int H, int W,           // 图像的高H 和宽 W
        int kernel_size,        // 最小值滤波核的边长
        data_type EXTREMUM, // 用于边缘 padding 的值
        bool use_min            // 是否用最小值滤波; false 的话就是最大值滤波
    ) {

    // 滤波核大小必须是正奇数
    assert(kernel_size > 0 and kernel_size & 1);

    // 获取中间参数
    int radius = (kernel_size - 1) >> 1;
    bool (*comp)(data_type, data_type) = use_min ? less_than: greater_than;
    EXTREMUM = use_min ? EXTREMUM: -EXTREMUM;

    // 对数据做 padding
    int H2 = H + 2 * radius;
    int W2 = W + 2 * radius;
    int pad_size = H2 * W2;
    data_type* padded_data = (data_type*)malloc(sizeof(data_type) * pad_size);
    for(int i = 0;i < pad_size; ++i) 
        padded_data[i] = EXTREMUM;
    for(int i = 0; i < H; ++i) 
        memcpy(padded_data + (i + radius) * W2 + radius, src + i * W, sizeof(data_type) * W);

    // 声明单调队列的参数
    int win_len = kernel_size;
    int front = 0;
    int back = 0;
    int capacity = kernel_size + 1;
    int* Q = (int*)malloc(sizeof(int) * kernel_size);

    // 先做水平方向的最小值滤波
    data_type* temp = (data_type*)malloc(sizeof(data_type) * pad_size);
    for(int i = 0;i < pad_size; ++i) temp[i] = EXTREMUM;
    for(int i = 0; i < H; ++i) {
        data_type* row_ptr = padded_data + (i + radius) * W2;
        data_type* res_ptr = temp + (i + radius) * W2 + radius;
        // 初始化单调队列
        data_type* data = row_ptr;
        front = back = 0;
        // 先放第一个元素
        back = (back + 1) % capacity;
        Q[back] = 0;
        // 接下来移动窗口, 记录窗口内的单调递减序列
        for(int j = 1; j < W2; ++j) {
            if(j >= kernel_size)
                res_ptr[j - kernel_size] = row_ptr[Q[(front + 1) % capacity]];
            // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
            const int pos = j;
            while(front != back) {
                int back_index = Q[back];
                if(comp(data[pos], data[back_index]))  // 如果队列中都比当前元素小, 停止
                    back = (back - 1 + capacity) % capacity;
                else break; // 否则, 把队列中大于当前元素的 popback 掉
            }
            // 当前元素的坐标放到这里
            back = (back + 1) % capacity;
            Q[back] = pos;
            // 如果当前维护的区间长度超出了窗口
            int front_next = (front + 1) % capacity;
            if(pos - Q[front_next] == win_len)
                front = front_next;
        }
        res_ptr[W2 - kernel_size] = row_ptr[Q[(front + 1) % capacity]];    
    }

    // 申请一块内存, 大小和 src 一致, 用于存储返回的结果(这里申请的内存是怎么释放的呢)
    data_type* res = (data_type*)malloc(sizeof(data_type) * H * W);

    // 准备竖直方向上的最小值滤波, 要更新一些参数
    win_len = kernel_size * W2;

    for(int i = 0; i < W; ++i) {
        data_type* col_ptr = temp + i + radius;
        data_type* res_ptr = res + i;
        // 重置单调队列的数据指针到这一列 
        data_type* data = col_ptr;
        front = back = 0;
        // 放第一个元素
        back = (back + 1) % capacity;
        Q[back] = 0;
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size) 
                res_ptr[(j - kernel_size) * W] = col_ptr[Q[(front + 1) % capacity]];
            // 注意起始位置是 W2 的倍数
            const int pos = j * W2;
            while(front != back) {
                int back_index = Q[back];
                if(comp(data[pos], data[back_index]))  // 如果队列中都比当前元素小, 停止
                    back = (back - 1 + capacity) % capacity;
                else break; // 否则, 把队列中大于当前元素的 popback 掉
            }
            // 当前元素的坐标放到这里
            back = (back + 1) % capacity;
            Q[back] = pos;
            // 如果当前维护的区间长度超出了窗口
            int front_next = (front + 1) % capacity;
            if(pos - Q[front_next] == win_len)
                front = front_next;
        }
        res_ptr[(H2 - kernel_size) * W] = col_ptr[Q[(front + 1) % capacity]];   
    }

    // 释放内存(如果是处理同一个分辨率的视频, 这三个可以放到外面, 不用每次处理一张图像就申请销毁一次, 太慢了)
    free(temp);
    free(Q);
    free(padded_data);

    return res;
}



































/**************************************** 动态规划方法 ***************************************
《A fast algorithm for local minimum and maximum filters on rectangular and octagonal kernels》
*/

void display1D(data_type* data, int length) {
    for(int i = 0; i < length; ++i)
        printf("%d  ", int(data[i]));
    printf("\n");
}

void display2D(data_type* data, int H, int W) {
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < W; ++j)
            printf("%d  ", int(data[i * W + j]));
        printf("\n");
    }
}

data_type min_element(data_type lhs, data_type rhs) {
    return lhs <= rhs ? lhs : rhs;
}

data_type max_element(data_type lhs, data_type rhs) {
    return lhs >= rhs ? lhs : rhs;
}

void swap(data_type* lhs, data_type* rhs) {
    data_type temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

void swap_pointer(data_type** lhs, data_type** rhs) {
    data_type* temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}


/*
    data_type image[12] = {1, 5, 10, 7, 9, 20, 4, 25, 12, 16, 18, 9};
    data_type* result = dynamic_min_filtering_1D(image, 12, 5, 255, true);
    display1D(result, 12);
    free(result);
*/
data_type* dynamic_min_filtering_1D(data_type* src, int W, int kernel_size, data_type EXTREMUM, bool use_min) {

    // 滤波核大小必须是正奇数
    assert(kernel_size > 0 and kernel_size & 1);

    // 最小核还是最大核
    data_type (*select)(data_type, data_type) = use_min ? min_element: max_element;
    EXTREMUM = use_min ? EXTREMUM: -EXTREMUM;  // 这个要不找个最大值, 遍历一遍, 有点麻烦

    // 对数据做 padding
    int radius = (kernel_size - 1) >> 1;
    int W2 = W + 2 * radius;                      // 先把前后最大值填充上
    int segment = ceil(W2 / double(kernel_size)); // 看最多能分成几段
    W2 = segment * kernel_size;                   // 重新得到被滤波的总长度, 为 kernel_size 的整数倍

    // padding 的主要操作, 分配空间
    data_type* padded_data = (data_type*)malloc(sizeof(data_type) * W2);
    for(int i = 0;i < W2; ++i) padded_data[i] = EXTREMUM;
    memcpy(padded_data + radius, src, sizeof(data_type) * W);


    // 准备一个结果
    data_type* result = (data_type*)malloc(sizeof(data_type) * W);

    // 开始滤波
    data_type* forward  = (data_type*)malloc(sizeof(data_type) * W2);
    data_type* backward = (data_type*)malloc(sizeof(data_type) * W2);

    for(int s = 0; s < segment; ++s) {
        // 找起始地址
        data_type* data     = padded_data + s * kernel_size;
        data_type* for_ptr  = forward     + s * kernel_size;
        data_type* back_ptr = backward    + s * kernel_size;
        // 记录前向最小值
        for_ptr[0] = data[0];
        for(int k = 1; k < kernel_size; ++k)
            for_ptr[k] = select(for_ptr[k - 1], data[k]);
        // 记录后向最小值
        back_ptr[kernel_size - 1] = data[kernel_size - 1];
        for(int k = kernel_size - 2; k >= 0; --k)
            back_ptr[k] =  select(back_ptr[k + 1], data[k]);
    }
 
    // 接下来, 计算每一个位置在往前数, 往后数 radius 个数的更小值
    for(int i = radius, i_max = radius + W; i < i_max; ++i)
        result[i - radius] = select(forward[i + radius], backward[i - radius]);

    if(false) {
        printf("segment = %d\n", segment);
        printf("W2 = %d\n", W2);
        printf("数据====>  "); display1D(padded_data, W2);
        printf("前向====>  "); display1D(forward, segment * kernel_size);
        printf("后向====>  "); display1D(backward, segment * kernel_size);
        printf("结果====>  "); 
    }

    free(backward);
    free(forward);
    free(padded_data);
    return result;
}



void violent_transpose2D(data_type* src, data_type* temp, int H, int W) {
    for(int i = 0;i < H; ++i)
        for(int j = 0; j < W; ++j)
            temp[j * H + i] = src[i * W + j];
}



void dynamic_min_filtering1D(data_type* buffer, data_type* res_ptr, int H, int W, int radius, int segment, data_type* forward, data_type* backward, data_type (*select)(data_type, data_type)) {
    int kernel_size = 2 * radius + 1;
    // segment 段分别算前向跟后向
    for(int s = 0; s < segment; ++s) {
        // 找起始地址
        data_type* data     = buffer      + s * kernel_size;
        data_type* for_ptr  = forward     + s * kernel_size;
        data_type* back_ptr = backward    + s * kernel_size;
        // 记录前向最小值
        for_ptr[0] = data[0];
        for(int k = 1; k < kernel_size; ++k)
            for_ptr[k] = select(for_ptr[k - 1], data[k]);
        // 记录后向最小值
        back_ptr[kernel_size - 1] = data[kernel_size - 1];
        for(int k = kernel_size - 2; k >= 0; --k)
            back_ptr[k] =  select(back_ptr[k + 1], data[k]);
    }
    // 接下来, 计算每一个位置在往前数, 往后数 radius 个数的更小值
    for(int i = radius, i_max = radius + W; i < i_max; ++i)
        res_ptr[i - radius] = select(forward[i + radius], backward[i - radius]);
}



// 如果是二维图像, 可以不用一次性 padding, 节约内存, 然后每次重新 padding, 我感觉可以, 之前写的程序还能继续优化
// 不推荐调用上面的 1D, 因为每调用一次, 就要分配一次内存, 速度太慢, 直接放到一起
data_type* dynamic_min_filtering(data_type* src, int H, int W, int kernel_size, data_type EXTREMUM, bool use_min) {
    // 滤波核大小必须是正奇数
    assert(kernel_size > 0 and kernel_size & 1);

    // 最小核还是最大核
    data_type (*select)(data_type, data_type) = use_min ? min_element: max_element;
    EXTREMUM = use_min ? EXTREMUM: -EXTREMUM;  // 这个要不找个最大值, 遍历一遍, 有点麻烦

    // 准备一个结果
    data_type* result = (data_type*)malloc(sizeof(data_type) * H * W);

    // 准备水平滤波的当前行的 padding 数据
    int radius  = (kernel_size - 1) >> 1;                    // 滤波核半径
    int W2      = H >= W ? H + 2 * radius: W + 2 * radius;   // 一行或者一列做填充后的长度   
    int segment = ceil(W2 / double(kernel_size));            // 看填充后, 需要几段 kernel_size 才可以覆盖
    W2          = segment * kernel_size;                     // 重新计算填充后的长度
    data_type* buffer = (data_type*)malloc(sizeof(data_type) * W2);  // 每一次, 当一维的数据做 padding, 在 padding 的数据上做最小值滤波
    for(int i = 0; i < W2; ++i) buffer[i] = EXTREMUM;        // padding 的值, 最小值滤波就填 INF; 最大值滤波就填 -INF

    // 准备前向后向数组
    data_type* forward  = (data_type*)malloc(sizeof(data_type) * W2);
    data_type* backward = (data_type*)malloc(sizeof(data_type) * W2);        

    // 先做水平方向上的滤波
    for(int i = 0; i < H; ++i) {
        // 把数据拷贝到缓冲区处理
        memcpy(buffer + radius, src + i * W, sizeof(data_type) * W);
        dynamic_min_filtering1D(buffer, result + i * W, H, W, radius, segment, forward, backward, select);
    }

    // 对第一次滤波结果 result, 转置结果放在 transpose_temp 上
    data_type* transpose_temp = (data_type*)malloc(sizeof(data_type) * H * W);
    violent_transpose2D(result, transpose_temp, H, W);
    
    // 第二次滤波, buffer 需要重新填充 padding, 防止第二次滤波的总长度更短——导致第一次的滤波结果成了 padding 值
    for(int i = 0; i < W2; ++i) buffer[i] = EXTREMUM;    
    for(int i = 0; i < W; ++i) {
        // 把数据拷贝到缓冲区处理(注意 H, W 全部颠倒)
        memcpy(buffer + radius, transpose_temp + i * H, sizeof(data_type) * H);
        dynamic_min_filtering1D(buffer, transpose_temp + i * H, W, H, radius, segment, forward, backward, select);
    }

    // 转置回去
    violent_transpose2D(transpose_temp, result, W, H);

    // 释放内存; 如果是视频处理同一个分辨率的图像, 就改成类, 在析构函数里析构
    free(transpose_temp);
    free(backward);
    free(forward);
    free(buffer);
    return result;
}


// int main() {

//     data_type image[15];
//     for(int i = 0; i < 15; ++i)
//         image[i] = i + 1;

//     data_type* result = dynamic_min_filtering(image, 3, 5, 3, 255, true);
//     display2D(result, 3, 5);
//     free(result);

//     // result = fast_min_filtering(image, 3, 5, 3, 255, true);
//     // display2D(result, 3, 5);
//     // free(result);

//     return 0;
// }