# Python
import os
import sys
import time
import ctypes
# 3rd party
import cv2
import numpy
import torch
from numpy.ctypeslib import ndpointer
# self
import torch_warp
import flow_viz


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)
add_to_save = lambda x: os.path.join(save_dir, x)



# 先读取图像
image1 = cv2.imread("./images/sintel/frame_0016.png")
image2 = cv2.imread("./images/sintel/frame_0017.png")
height, width, channel = image1.shape
make_show              = True if (height * width < 1024 * 768) else False

# 获取 image1 → image2 的光流
use_flow_cache      = True
save_flow_cache     = True
forward_flow_cache  = "./images/sintel/forward_flow.npy"
backward_flow_cache = "./images/sintel/backward_flow.npy"
if (use_flow_cache):
	forward_flow    = numpy.load(forward_flow_cache)
	backward_flow   = numpy.load(backward_flow_cache)
else:
	forward_flow, backward_flow = flow_viz.compute_optical_flow(image1, image2)

# 如果确认缓存光流, 而且大小不是很大, 缓存之
if (save_flow_cache and make_show):
	numpy.save(forward_flow_cache,  forward_flow)
	numpy.save(backward_flow_cache, backward_flow)


# 可视化光流
forward_flow_visualize  = flow_viz.flow_to_image(forward_flow)[:, :, ::-1] # [:, :, ::-1] 是为了 opencv 显示 BGR 序
backward_flow_visualize = flow_viz.flow_to_image(backward_flow)[:, :, ::-1]
cv_write(add_to_save("forward_flow_visualize.png"),  forward_flow_visualize)
cv_write(add_to_save("backward_flow_visualize.png"), backward_flow_visualize)
cv_show(numpy.concatenate([forward_flow_visualize, backward_flow_visualize], axis=0))


# 数据都改成 torch, 放到 GPU 上
image1 = torch.as_tensor(image1).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
image2 = torch.as_tensor(image2).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
forward_flow  = torch.as_tensor(forward_flow).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
backward_flow = torch.as_tensor(backward_flow).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()


# 提前选一个结果
warp_2to1_using_crane = torch.zeros(image1.shape, dtype=torch.uint8).cuda()
warp_1to2_using_crane = torch.zeros(image2.shape, dtype=torch.uint8).cuda()
# 需要进入 torch_interface 目录, 运行 python setup.py install, 安装 fast_optical_flow
import fast_optical_flow
fast_optical_flow.warp(warp_2to1_using_crane, image2, forward_flow)
fast_optical_flow.warp(warp_1to2_using_crane, image1, backward_flow)
# 做前后 warp
warp_2to1_using_crane = warp_2to1_using_crane.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
warp_1to2_using_crane = warp_1to2_using_crane.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
cv_show(numpy.concatenate([warp_2to1_using_crane, warp_1to2_using_crane], axis=0))

# 快速计算遮挡
dis_threshold = 2.0
forward_occulusion = torch.zeros((1, height, width), dtype=torch.uint8).cuda()
fast_optical_flow.lrcheck(forward_occulusion, forward_flow, backward_flow, dis_threshold)
forward_occulusion = forward_occulusion.detach().cpu().squeeze(0).numpy()

backward_occulusion = torch.zeros((1, height, width), dtype=torch.uint8).cuda()
fast_optical_flow.lrcheck(backward_occulusion, backward_flow, forward_flow, dis_threshold)
backward_occulusion = backward_occulusion.detach().cpu().squeeze(0).numpy()
cv_show(numpy.concatenate([forward_occulusion, backward_occulusion], axis=0))
