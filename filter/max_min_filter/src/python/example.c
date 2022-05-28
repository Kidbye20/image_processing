#include <math.h>
#include <stdio.h>
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