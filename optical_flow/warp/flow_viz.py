# 3rd party
import cv2
import numpy




# 设置一个全局变量, 不需要重复加载 onnx 文件
infer_task = None



def compute_optical_flow(image1, image2, do_upsample=True, return_img=False):
    height, width, _ = image1.shape
    assert image1.shape == image2.shape, "image1 和 image2 的形状不同"
    make_pad     = False
    # 如果是高分辨率的图像
    if (height * width > 1024 * 768):
        make_resize   = True
        lowres_height = 768
        lowres_width  = 1024
    else:
        make_resize   = False
        # GMFlowNet 只支持边长为 8 倍数的图像, 所以需要做 padding
        if (int(height / 8) == 0 and int(width / 8) == 0): 
            pass
        else:
            # 不是 8 的倍数, 需要做 padding, 但这也必须保证是 2 的倍数
            make_pad = True
            height_2, width_2 = 8 * (int(height / 8) + 1), 8 * (int(width / 8) + 1)
            h_pad, w_pad = int((height_2 - height) / 2), int((width_2 - width) / 2)

    # 做缩放和
    if (make_resize):
        lowres_image1 = cv2.resize(image1, (lowres_width, lowres_height), cv2.INTER_LANCZOS4)
        lowres_image2 = cv2.resize(image2, (lowres_width, lowres_height), cv2.INTER_LANCZOS4)
    elif (make_pad):
        lowres_image1 = numpy.pad(image1, [(h_pad, h_pad), (w_pad, w_pad), (0, 0)], mode="reflect")
        lowres_image2 = numpy.pad(image2, [(h_pad, h_pad), (w_pad, w_pad), (0, 0)], mode="reflect")

    # 设置转换函数, 从 numpy、uint8、BGR序、HWC → numpy、float32、RGB序、1CHW
    def convert_to_tensor(x):
        x_tensor = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x_tensor = x_tensor.transpose(2, 0, 1)
        x_tensor = numpy.ascontiguousarray(x_tensor)
        x_tensor = numpy.expand_dims(x_tensor, axis=0)
        return x_tensor.astype("float32")

    # 把原始图像转换成 B x C x H x W 的内存格式
    image1_tensor = convert_to_tensor(lowres_image1)
    image2_tensor = convert_to_tensor(lowres_image2)
    # print("image1  :  {}\nimage2  :  {}\n".format(image1_tensor.shape, image2_tensor.shape))

    # 加载光流的 ONNX 模型
    global infer_task
    if (infer_task is None):
        import onnxruntime
        onnx_file  = "GMFlowNet.onnx"
        infer_task = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    # 开始推理
    [forward_flow]  = infer_task.run(["flow"], {"image1": image1_tensor, "image2": image2_tensor})
    [backward_flow] = infer_task.run(["flow"], {"image1": image2_tensor, "image2": image1_tensor})
    forward_flow    = numpy.ascontiguousarray(forward_flow[0].transpose(1, 2, 0))
    backward_flow   = numpy.ascontiguousarray(backward_flow[0].transpose(1, 2, 0))

    # 如果做了放缩, 则需要对图像做放缩, 同时光流要乘以放缩倍率
    if (make_resize and do_upsample):
        print("光流上采样, 需要乘以倍率")
        h_ratio = float(height / forward_flow.shape[0])
        w_ratio = float(width  / forward_flow.shape[1])
        forward_flow  = cv2.resize(forward_flow, (width, height), cv2.INTER_LINEAR)
        backward_flow = cv2.resize(backward_flow, (width, height), cv2.INTER_LINEAR)
        print("h_ratio  {}\nw_ratio  {}".format(h_ratio, w_ratio))
        forward_flow[:, :, 0]  *= w_ratio
        forward_flow[:, :, 1]  *= h_ratio
        backward_flow[:, :, 0] *= w_ratio
        backward_flow[:, :, 1] *= h_ratio
    elif (make_pad):
        forward_flow  = forward_flow[h_pad: h_pad + height, w_pad: w_pad + width].copy()
        backward_flow = backward_flow[h_pad: h_pad + height, w_pad: w_pad + width].copy()
    # 清下内存
    del image1_tensor, image2_tensor

    # 返回
    return (forward_flow, backward_flow) if (not return_img) else (forward_flow, backward_flow, lowres_image1, lowres_image2)





def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        numpy.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = numpy.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = numpy.floor(255*numpy.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - numpy.floor(255*numpy.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = numpy.floor(255*numpy.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - numpy.floor(255*numpy.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = numpy.floor(255*numpy.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - numpy.floor(255*numpy.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (numpy.ndarray): Input horizontal flow of shape [H,W]
        v (numpy.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        numpy.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = numpy.zeros((u.shape[0], u.shape[1], 3), numpy.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = numpy.sqrt(numpy.square(u) + numpy.square(v))
    a = numpy.arctan2(-v, -u)/numpy.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = numpy.floor(fk).astype(numpy.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = numpy.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (numpy.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        numpy.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = numpy.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = numpy.sqrt(numpy.square(u) + numpy.square(v))
    rad_max = numpy.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)