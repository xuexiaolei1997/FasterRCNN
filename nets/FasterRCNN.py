import torch.nn as nn

from nets.classifier import Resnet50RoIHead
from nets.resnet import resnet50
from nets.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, mode="training", feat_stride=16, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2],
                 backbone="resnet50"):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        if backbone == 'resnet50':
            self.extractor, classifier = resnet50()
            self.rpn = RegionProposalNetwork(1024, 512, ratios=ratios, anchor_scales=anchor_scales,
                                             feat_stride=feat_stride, mode=mode)
            self.head = Resnet50RoIHead(n_class=num_classes + 1, roi_size=14, spatial_scale=1, classifier=classifier)

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        base_feature = self.extractor(x)
        _, _, rois, roi_indices, _ = self.rpn(base_feature, img_size, scale)
        roi_cls_locs, roi_scores = self.head(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
