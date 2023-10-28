import torch
from torch.nn import Module, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Parameter
import torch.nn.functional as F
import torchvision

class ScaledL2Norm(Module):
    """https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/nn/scaled_l2_norm.py"""
    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()
    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1)
        normalized_x = x / norm
        return (normalized_x * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3))

class ConvBNScaleReLU(Module):
    '''
    BatchNorm(affine=True) in PyTorch == BatchNorm + Scale in Caffe
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    https://github.com/BVLC/caffe/blob/master/include/caffe/layers/batch_norm_layer.hpp#L28
    '''
    def __init__(self, in_c, out_c, kernel_size, padding=0, stride=1, bias=False):
        super(ConvBNScaleReLU, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn   = BatchNorm2d(num_features=out_c, affine=True)
    def forward(self, x):
        return torch.relu_(self.bn(self.conv(x)))

class InceptionA(Module):
    def __init__(self, in_c, pool_proj_c):
        super(InceptionA, self).__init__()
        self.inception_a_1x1 = ConvBNScaleReLU(in_c, 64, 1)
        self.inception_a_5x5_reduce = ConvBNScaleReLU(in_c, 48, 1)
        self.inception_a_5x5 = ConvBNScaleReLU(48, 64, 5, padding=2)
        self.inception_a_3x3_reduce = ConvBNScaleReLU(in_c, 64, 1)
        self.inception_a_3x3_1 = ConvBNScaleReLU(64, 96, 3, padding=1)
        self.inception_a_3x3_2 = ConvBNScaleReLU(96, 96, 3, padding=1)
        self.inception_a_pool = AvgPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_a_pool_proj = ConvBNScaleReLU(in_c, pool_proj_c, 1)
    def forward(self, x):
        inception_a_1x1 = self.inception_a_1x1(x)
        inception_a_5x5_reduce = self.inception_a_5x5_reduce(x)
        inception_a_5x5 = self.inception_a_5x5(inception_a_5x5_reduce)
        inception_a_3x3_reduce = self.inception_a_3x3_reduce(x)
        inception_a_3x3_1 = self.inception_a_3x3_1(inception_a_3x3_reduce)
        inception_a_3x3_2 = self.inception_a_3x3_2(inception_a_3x3_1)
        inception_a_pool = self.inception_a_pool(x)
        inception_a_pool_proj = self.inception_a_pool_proj(inception_a_pool)
        return torch.concat([inception_a_1x1, inception_a_5x5, inception_a_3x3_2, inception_a_pool_proj], dim=1)

class InceptionB(Module):
    def __init__(self, in_c, intermediate_c, out_c=192):
        super(InceptionB, self).__init__()
        self.inception_b_1x1_2 = ConvBNScaleReLU(in_c, out_c, 1)
        self.inception_b_1x7_reduce = ConvBNScaleReLU(in_c, intermediate_c, 1)
        self.inception_b_1x7 = ConvBNScaleReLU(intermediate_c, intermediate_c, (1, 7), padding=(0, 3))
        self.inception_b_7x1 = ConvBNScaleReLU(intermediate_c, out_c, (7, 1), padding=(3, 0))
        self.inception_b_7x1_reduce = ConvBNScaleReLU(in_c, intermediate_c, 1)
        self.inception_b_7x1_2 = ConvBNScaleReLU(intermediate_c, intermediate_c, (7, 1), padding=(3, 0))
        self.inception_b_1x7_2 = ConvBNScaleReLU(intermediate_c, intermediate_c, (1, 7), padding=(0, 3))
        self.inception_b_7x1_3 = ConvBNScaleReLU(intermediate_c, intermediate_c, (7, 1), padding=(3, 0))
        self.inception_b_1x7_3 = ConvBNScaleReLU(intermediate_c, out_c, (1, 7), padding=(0, 3))
        self.inception_b_pool_ave = AvgPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.inception_b_1x1 = ConvBNScaleReLU(in_c, out_c, 1)
    def forward(self, x):
        inception_b_1x1_2 = self.inception_b_1x1_2(x)
        inception_b_1x7_reduce = self.inception_b_1x7_reduce(x)
        inception_b_1x7 = self.inception_b_1x7(inception_b_1x7_reduce)
        inception_b_7x1 = self.inception_b_7x1(inception_b_1x7)
        inception_b_7x1_reduce = self.inception_b_7x1_reduce(x)
        inception_b_7x1_2 = self.inception_b_7x1_2(inception_b_7x1_reduce)
        inception_b_1x7_2 = self.inception_b_1x7_2(inception_b_7x1_2)
        inception_b_7x1_3 = self.inception_b_7x1_3(inception_b_1x7_2)
        inception_b_1x7_3 = self.inception_b_1x7_3(inception_b_7x1_3)
        inception_b_pool_ave = self.inception_b_pool_ave(x)
        inception_b_1x1 = self.inception_b_1x1(inception_b_pool_ave)
        return torch.concat([inception_b_1x1_2, inception_b_7x1, inception_b_1x7_3, inception_b_1x1], dim=1)

def gen_prior_boxes(prior_box_param):
    # get params
    min_sizes = torch.tensor(prior_box_param['min_sizes'])
    ars = torch.tensor(prior_box_param['aspect_ratios'])
    var = torch.tensor(prior_box_param['variance'])
    offset_ws = torch.tensor(prior_box_param['offset_ws'])
    offset_hs = torch.tensor(prior_box_param['offset_hs'])
    image_shape = prior_box_param['image_shape']
    feature_map_shape = prior_box_param['feature_map_shape']
    # compute steps
    image_h = image_shape[2]
    image_w = image_shape[3]
    layer_h = feature_map_shape[2]
    layer_w = feature_map_shape[3]
    step_h = image_h / layer_h
    step_w = image_w / layer_w
    # create a list of centers coordinates
    hrange = torch.arange(layer_h)
    wrange = torch.arange(layer_w)
    shift_h, shift_w = torch.meshgrid(hrange, wrange, indexing='ij')
    shift_h = shift_h.reshape(-1)
    shift_w = shift_w.reshape(-1)
    shifts = torch.hstack([shift_w[:, None], shift_h[:, None]])
    # apply offsets to each center
    offset_h, offset_w = torch.meshgrid(offset_hs, offset_ws, indexing='xy')
    offset_h = offset_h.reshape(-1)
    offset_w = offset_w.reshape(-1)
    offsets = torch.hstack([offset_w[:, None], offset_h[:, None]])
    centers = shifts[:, None, :] + offsets[None, :, :]
    centers = centers.reshape(-1, 2)
    # zoom up to the image size
    centers = centers * torch.tensor([step_w, step_h])
    # apply the aspect_ratios and min_sizes
    box_widths = (min_sizes[:, None] * torch.sqrt(ars)[None, :]).reshape(-1)
    box_heights = (min_sizes[:, None] / torch.sqrt(ars)[None, :]).reshape(-1)
    # convert to xmin, ymin, xmax, ymax
    xmins = (centers[:, 0][:, None] - box_widths[None, :]).reshape(-1, 1)
    ymins = (centers[:, 1][:, None] - box_heights[None, :]).reshape(-1, 1)
    xmaxs = (centers[:, 0][:, None] + box_widths[None, :]).reshape(-1, 1)
    ymaxs = (centers[:, 1][:, None] + box_heights[None, :]).reshape(-1, 1)
    anchors = torch.hstack([xmins, ymins, xmaxs, ymaxs])
    variance = torch.ones_like(anchors) * var
    return anchors, variance

def nms(boxes, scores, k, score_threshold, nms_threshold):
    values, indices = torch.topk(scores[:, 1], k)
    indices = indices[values >  score_threshold]
    boxes = boxes[indices]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # calculate area of every block in P
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # initialise an empty list for
    # filtered prediction boxes
    keep_inds = []
    idxs = torch.arange(boxes.shape[0]).to(boxes.device)
    while len(idxs) > 0:
        idx = idxs[0]
        # push S in filtered predictions list
        keep_inds.append(idx)
        idxs = idxs[1:]
        if len(idxs) == 0:
            break
        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = x1[idxs]
        xx2 = x2[idxs]
        yy1 = y1[idxs]
        yy2 = y2[idxs]
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        # find height and width of the intersection boxes
        w = xx2 - xx1 + 1
        h = yy2 - yy1 + 1
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        # find the intersection area
        inter = w * h
        # find the areas of BBoxes according the indices in order
        rem_areas = areas[idxs]
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        # find the IoU of every prediction in P with S
        IoU = inter / union
        # keep the boxes with IoU less than thresh_iou
        mask = IoU <= nms_threshold
        idxs = idxs[mask]
    return indices[keep_inds]

class DetectionOutput(Module):
    def __init__(self, prior_box_param, detection_output_param, device=None):
        super(DetectionOutput, self).__init__()
        self.device = device
        self.prior_boxes, self.variances = gen_prior_boxes(prior_box_param)
        self.prior_boxes = self.prior_boxes.to(device)
        self.variances = self.variances.to(device)
        self.nms_top_k = detection_output_param['nms_top_k']
        self.nms_threshold = detection_output_param['nms_threshold']
        self.keep_top_k = detection_output_param['keep_top_k']
        self.conf_threshold = detection_output_param['conf_threshold']
    def forward(self, locs, conf_scores, state_scores):
        # handle location
        prior_center_xs = (self.prior_boxes[:, 0] + self.prior_boxes[:, 2]) / 2
        prior_center_ys = (self.prior_boxes[:, 1] + self.prior_boxes[:, 3]) / 2
        prior_widths = self.prior_boxes[:, 2] - self.prior_boxes[:, 0]
        prior_heights = self.prior_boxes[:, 3] - self.prior_boxes[:, 1]
        print(self.variances.shape, locs.shape, prior_widths.shape, prior_heights.shape)
        decode_bbox_center_xs = self.variances[:, 0] * locs[:, :, 0] * prior_widths + prior_center_xs
        decode_bbox_center_ys = self.variances[:, 1] * locs[:, :, 1] * prior_heights + prior_center_ys
        decode_bbox_widths = torch.exp(self.variances[:, 2] * locs[:, :, 2]) * prior_widths
        decode_bbox_heights = torch.exp(self.variances[:, 3] * locs[:, :, 3]) * prior_heights
        decode_bboxes_xmins = decode_bbox_center_xs - decode_bbox_widths / 2
        decode_bboxes_ymins = decode_bbox_center_ys - decode_bbox_heights / 2
        decode_bboxes_xmaxs = decode_bbox_center_xs + decode_bbox_widths / 2
        decode_bboxes_ymaxs = decode_bbox_center_ys + decode_bbox_heights / 2
        decode_bboxes = torch.concat([decode_bboxes_xmins.unsqueeze(2), decode_bboxes_ymins.unsqueeze(2), decode_bboxes_xmaxs.unsqueeze(2), decode_bboxes_ymaxs.unsqueeze(2)], dim=2)
        # make detections based on conf scores
        detections = []
        for i in range(decode_bboxes.shape[0]):
            indices = nms(decode_bboxes[i], conf_scores[i], self.nms_top_k, self.conf_threshold, self.nms_threshold)
            if len(indices) < self.keep_top_k:
                indices = indices[:self.keep_top_k]
            detections.append((decode_bboxes[:, indices], conf_scores[:, indices], state_scores[:, indices]))
        return detections

class TLSSD(Module):
    def __init__(self, prior_box_param, detection_output_param, device=None):
        super(TLSSD, self).__init__()
        self.conv1_3x3_s2 = ConvBNScaleReLU(3, 32, 3, stride=2) # shape reduced to (1, 32, 255, 1023)
        self.conv2_3x3_s1 = ConvBNScaleReLU(32, 32, 3) # shape (1, 32, 253, 1021)
        self.conv3_3x3_s1 = ConvBNScaleReLU(32, 64, 3, padding=1) # shape (1, 64, 253, 1021)
        self.pool1_3x3_s2 = MaxPool2d(3, stride=2, ceil_mode=True) # shape (1, 64, 126, 510)
        self.conv4_3x3_reduce = ConvBNScaleReLU(64, 80, 1) # shape (1, 80, 126, 510)
        self.conv4_3x3 = ConvBNScaleReLU(80, 192, 3) # shape (1, 192, 124, 508)
        self.pool2_3x3_s2 = MaxPool2d(3, stride=2, ceil_mode=True) # shape (1, 192, 62, 254)
        self.inception_a1 = InceptionA(192, 32) # shape (1, 256, 62, 254)
        self.inception_a2 = InceptionA(256, 64) # shape (1, 288, 62, 254)
        self.inception_a3 = InceptionA(288, 64) # shape (1, 288, 62, 254)
        self.reduction_a_pool = MaxPool2d(3, stride=2, ceil_mode=True) # from inception_a3, shape (1, 288, 31, 127)
        self.reduction_a_3x3 = ConvBNScaleReLU(288, 384, 3, stride=2, padding=1) # from inception_a3, shape (1, 384, 31, 127)
        self.reduction_a_3x3_2_reduce = ConvBNScaleReLU(288, 64, 1) # from inception_a3, shape (1, 64, 61, 253)
        self.reduction_a_3x3_2 = ConvBNScaleReLU(64, 96, 3, padding=1) # from reduction_a_3x3_2_reduce, shape (1, 96, 61, 253)
        self.reduction_a_3x3_3 = ConvBNScaleReLU(96, 96, 3, padding=1, stride=2) # from reduction_a_3x3_2, shape (1, 96, 31, 127)
        self.inception_b1 = InceptionB(768, 128) # from [reductiona_pool, reduction_a_3x3, reduction_a_3x3_3] whose c is 288 + 384 + 96 = 768, shape (1, 768, 31, 127)
        self.inception_b2 = InceptionB(768, 160) # shape (1, 768, 31, 127)
        self.inception_b3 = InceptionB(768, 160) # shape (1, 768, 31, 127)
        self.inception_b4 = InceptionB(768, 192) # shape (1, 768, 31, 127)
        self.inception_b4_concat_norm = ScaledL2Norm(768, 20) # shape (1, 768, 31, 127)
        # location boxes
        self.inception_b4_concat_norm_mbox_loc = Conv2d(768, 28, 1, bias=True) # shape (1, 28, 31, 127) 7 * 4
        # conf scores
        self.inception_b4_concat_norm_mbox_conf = Conv2d(768, 14, 1, bias=True) # shape (1, 14, 31, 127) 7 * 2
        # states classification
        self.inception_b4_concat_norm_mbox_state = Conv2d(768, 42, 1, bias=True) # shape (1, 42, 31, 127) 7 * 6
        self.detection_out = DetectionOutput(prior_box_param, detection_output_param, device)
    def forward(self, x):
        assert x.shape == (1, 3, 512, 2048)
        conv1_3x3_s2 = self.conv1_3x3_s2(x)
        conv2_3x3_s1 = self.conv2_3x3_s1(conv1_3x3_s2)
        conv3_3x3_s1 = self.conv3_3x3_s1(conv2_3x3_s1)
        pool1_3x3_s2 = self.pool1_3x3_s2(conv3_3x3_s1)
        conv4_3x3_reduce = self.conv4_3x3_reduce(pool1_3x3_s2)
        conv4_3x3 = self.conv4_3x3(conv4_3x3_reduce)
        pool2_3x3_s2 = self.pool2_3x3_s2(conv4_3x3)
        inception_a1 = self.inception_a1(pool2_3x3_s2)
        inception_a2 = self.inception_a2(inception_a1)
        inception_a3 = self.inception_a3(inception_a2)
        reduction_a_pool = self.reduction_a_pool(inception_a3)
        reduction_a_3x3 = self.reduction_a_3x3(inception_a3)
        reduction_a_3x3_2_reduce = self.reduction_a_3x3_2_reduce(inception_a3)
        reduction_a_3x3_2 = self.reduction_a_3x3_2(reduction_a_3x3_2_reduce)
        reduction_a_3x3_3 = self.reduction_a_3x3_3(reduction_a_3x3_2)
        reduction_a_concat = torch.concat([reduction_a_pool, reduction_a_3x3, reduction_a_3x3_3], dim=1)
        inception_b1 = self.inception_b1(reduction_a_concat)
        inception_b2 = self.inception_b2(inception_b1)
        inception_b3 = self.inception_b3(inception_b2)
        inception_b4 = self.inception_b4(inception_b3)
        inception_b4_concat_norm = self.inception_b4_concat_norm(inception_b4)
        inception_b4_concat_norm_mbox_loc = self.inception_b4_concat_norm_mbox_loc(inception_b4_concat_norm)
        inception_b4_concat_norm_mbox_loc_perm = inception_b4_concat_norm_mbox_loc.permute(0, 2, 3, 1)
        mbox_loc = inception_b4_concat_norm_mbox_loc_perm.reshape(inception_b4_concat_norm_mbox_loc_perm.shape[0], -1, 4)
        inception_b4_concat_norm_mbox_conf = self.inception_b4_concat_norm_mbox_conf(inception_b4_concat_norm)
        mbox_conf_reshape = inception_b4_concat_norm_mbox_conf.permute(0, 2, 3, 1).reshape(inception_b4_concat_norm_mbox_conf.shape[0], -1, 2)
        mbox_conf_softmax = F.softmax(mbox_conf_reshape, dim=2)
        inception_b4_concat_norm_mbox_state = self.inception_b4_concat_norm_mbox_state(inception_b4_concat_norm)
        inception_b4_concat_norm_mbox_state_perm = inception_b4_concat_norm_mbox_state.permute(0, 2, 3, 1)
        mbox_state = inception_b4_concat_norm_mbox_state_perm.reshape(inception_b4_concat_norm_mbox_state_perm.shape[0], -1, 6)
        mbox_state_sigmoid = torch.sigmoid(mbox_state)
        return self.detection_out(mbox_loc, mbox_conf_softmax, mbox_state_sigmoid)

prior_box_param = {
    'min_sizes': [7.0, 10.0, 15.0, 25.0, 35.0, 50.0, 70.0],
    'aspect_ratios': [0.3],
    'offset_ws': [0.5],
    'offset_hs': [0.5],
    'image_shape': [1, 3, 512, 2048],
    'feature_map_shape': [1, 42, 31, 127],
    'variance': [0.1]
}
detection_output_param = {
    'nms_top_k': 400,
    'nms_threshold': 0.35,
    'keep_top_k': 200,
    'conf_threshold': 0.01
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tlssd = TLSSD(prior_box_param, detection_output_param, device).to(device)
tlssd.load_state_dict(torch.load('tlssd.pth'))
tlssd.eval()
with torch.no_grad():
    x = torch.zeros(1, 3, 512, 2048)
    image = torchvision.io.read_image('test.jpg') - 60
    x[0][:, :, :1920] += image[:, 100:612, :]
    results = tlssd(x.to(device))
    print(results)
