


from torchvision.ops import roi_pool
def roi_pool(
    # input: Tensor, 输入的图片，格式[N,C,H,W]
    # boxes: Tensor, 要拿出来的区域，格式 [K,(x1, y1, x2, y2)]
    # output_size: BroadcastingList2[int], #pool后的大小，Tuple[int, int]
    # spatial_scale: float = 1.0, 将输入坐标映射到box坐标的尺度因子. 默认: 1.0
):
    pass
#返回 Tensor[K, C, output_size[0], output_size[1]]

def nms(boxes, scores, iou_threshold):
    pass
# boxes (Tensor[N, 4])) – bounding boxes坐标. 格式：(x1, y1, x2, y2)
# scores (Tensor[N]) – bounding boxes得分
# iou_threshold (float) – IoU过滤阈值
#keep :NMS过滤后的bouding boxes索引（降序排列）