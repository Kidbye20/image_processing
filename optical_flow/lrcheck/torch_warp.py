# 3rd party
import numpy
import torch


def __warp(x, flo):
	"""
	warp an image/tensor (im2) back to im1, according to the optical flow
	x: [B, C, H, W] (im2)
	flo: [B, 2, H, W] flow
	"""
	B, C, H, W = x.size()
	# mesh grid 
	xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
	yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
	xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
	yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
	grid = torch.cat((xx, yy), 1).float()
	vgrid = grid + flo # B,2,H,W
	#图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标
	vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0 
	#取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
	vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0 #取出光流u这个维度，同上
	vgrid = vgrid.permute(0, 2, 3, 1)#from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
	output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)

	mask = torch.ones(x.size())
	mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)
	mask[mask < 0.9999] = 0
	mask[mask > 0] = 1

	return output * mask


def warp(x, flow):
	# numpy → Pytorch
	torch_x = torch.as_tensor(x.astype("float32")).div(255).permute(2, 0, 1).unsqueeze(0).contiguous()
	torch_flow = torch.as_tensor(flow).permute(2, 0, 1).unsqueeze(0).contiguous()
	# Torch grid_sample
	warped = __warp(torch_x, torch_flow)
	# Pytorch → numpy
	return warped.squeeze(0).mul(255).permute(1, 2, 0).numpy().astype("uint8")
