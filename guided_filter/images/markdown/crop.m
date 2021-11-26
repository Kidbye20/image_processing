clc;
clear all;

origin = imread("../output/comparison_detail_enhancement.png");
[M, N, C] = size(origin);

N = N / 3;

x = 1270;
y = 400;
width = 60;
height = 60;

lhs = imcrop(origin, [x, y, width, height]);
rhs = imcrop(origin, [N + x, y, width, height]);
figure; imshow([lhs, rhs]);

imwrite([lhs, rhs], "./comparison_detail_enhancement_crop.png");
