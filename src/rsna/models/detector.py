"""Box detectors from torchvision (Faster R-CNN, RetinaNet, FCOS, SSD) on a pretrained timm backbone with an FPN
(SSD: extra down-sampling blocks instead of an FPN, NMS 0.45).

One class (Lung Opacity, label 1). Images are float (1, H, W) in [0, 1]; torchvision's transform normalizes them
with the backbone's channel-mean mean/std and keeps them at `size` (no resizing to 800 px).
Anchor sizes are torchvision's defaults scaled by size / 512: 16-256 px at 256, which covers the boxes (10-235 px).
The backbone and FPN run in bfloat16, the heads in float32 (box decoding in bfloat16 would lose pixels).
backbone="coco" uses torchvision's COCO-pretrained ResNet-50 FPN detector instead; the gray image is repeated to 3 channels.
"""
from collections import OrderedDict
from functools import partial

import timm
import torch
import torch.nn as nn
from torchvision.models.detection import FCOS, FasterRCNN, RetinaNet
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, fcos_resnet50_fpn, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, LastLevelP6P7

ARCHS = ["fasterrcnn", "retinanet", "fcos", "ssd"]


class Bf16(nn.Module):
    """Runs a backbone (+ FPN) under bfloat16 autocast and returns its feature maps as float32."""

    def __init__(self, body: nn.Module):
        super().__init__()
        self.body, self.out_channels = body, body.out_channels

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.body(x)
        return OrderedDict((k, v.float()) for k, v in features.items())


class TimmFPN(nn.Module):
    """timm feature maps at the given strides -> 256-channel FPN levels named "0", "1", ... (+ extra levels)."""

    def __init__(self, name: str, strides, extra_blocks, pretrained: bool = True, drop_path: float = 0.0):
        super().__init__()
        kwargs = dict(pretrained=pretrained, features_only=True, in_chans=1)
        reductions = timm.create_model(name.split(".")[0], pretrained=False, features_only=True).feature_info.reduction()
        kwargs["out_indices"] = [reductions.index(s) for s in strides]
        try:
            self.body = timm.create_model(name, drop_path_rate=drop_path, **kwargs)
        except TypeError:  # DenseNet has no drop path
            self.body = timm.create_model(name, **kwargs)
        self.fpn = FeaturePyramidNetwork(self.body.feature_info.channels(), 256, extra_blocks=extra_blocks)
        self.out_channels = 256

    def forward(self, x):
        return self.fpn(OrderedDict((str(i), f) for i, f in enumerate(self.body(x))))


class L2Norm(nn.Module):
    """SSD's L2 normalization: each position's feature vector scaled to unit length, times a learned per-channel scale."""

    def __init__(self, channels: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), scale))

    def forward(self, x):
        return nn.functional.normalize(x, dim=1) * self.weight[None, :, None, None]


class TimmSSD(nn.Module):
    """SSD feature maps: timm features at strides 8, 16, 32, then extra blocks that halve the map down to 1 x 1
    (strides 64, 128, 256 at 256 px), as in the original SSD (no FPN). The timm maps are L2-normalized, as SSD does
    for VGG's conv4_3, because their scale differs a lot between backbones (ConvNeXt's is large)."""

    def __init__(self, name: str, pretrained: bool = True, drop_path: float = 0.0):
        super().__init__()
        self.body = TimmFPN(name, (8, 16, 32), None, pretrained, drop_path).body  # reuse the timm setup, drop the FPN
        channels = self.body.feature_info.channels()
        self.norms = nn.ModuleList(L2Norm(c) for c in channels)
        self.extras = nn.ModuleList()
        for _ in range(3):
            self.extras.append(nn.Sequential(
                nn.Conv2d(channels[-1], 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)))
            channels.append(256)
        self.out_channels = channels

    def forward(self, x):
        features = [norm(f) for norm, f in zip(self.norms, self.body(x))]
        for block in self.extras:
            features.append(block(features[-1]))
        return OrderedDict((str(i), f) for i, f in enumerate(features))


def anchor_sizes(arch: str, size: int):
    """One size per FPN level (3 octave scales for RetinaNet), torchvision's defaults scaled by size / 512."""
    f = size / 512
    if arch == "fcos":
        return tuple((round(s * f),) for s in (8, 16, 32, 64, 128))
    if arch == "retinanet":
        return tuple(tuple(round(s * f * 2 ** (i / 3)) for i in range(3)) for s in (32, 64, 128, 256, 512))
    return tuple((round(s * f),) for s in (32, 64, 128, 256, 512))


def build_detector(arch: str, backbone: str, size: int, pretrained: bool = True, drop_path: float = 0.0):
    """Returns (model, in_chans); feed the model a list of (in_chans, size, size) float images."""
    if arch == "ssd":
        body = TimmSSD(backbone, pretrained, drop_path)
        cfg = body.body.pretrained_cfg
        # default box scales 0.07-0.9 of the image (18-230 px at 256), SSD300's aspect ratios
        anchors = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]], min_ratio=0.07, max_ratio=0.9)
        head = SSDHead(body.out_channels, anchors.num_anchors_per_location(), 2)
        return SSD(Bf16(body), anchors, (size, size), 2, image_mean=[sum(cfg["mean"]) / 3],
                   image_std=[sum(cfg["std"]) / 3], head=head, score_thresh=0.01, detections_per_img=20), 1

    sizes = anchor_sizes(arch, size)
    ratios = ((1.0,),) * 5 if arch == "fcos" else ((0.5, 1.0, 2.0),) * 5
    anchors = AnchorGenerator(sizes, ratios)
    common = dict(min_size=size, max_size=size)
    score = (dict(box_score_thresh=0.01, box_detections_per_img=20) if arch == "fasterrcnn"
             else dict(score_thresh=0.01, detections_per_img=20))

    if backbone == "coco":  # 3-channel ImageNet normalization (torchvision's default), new 1-class output layer
        build = {"fasterrcnn": fasterrcnn_resnet50_fpn_v2, "retinanet": retinanet_resnet50_fpn_v2, "fcos": fcos_resnet50_fpn}[arch]
        anchor_arg = {"fasterrcnn": "rpn_anchor_generator"}.get(arch, "anchor_generator")
        model = build(weights="COCO_V1", **common, **score, **{anchor_arg: anchors})
        if arch == "fasterrcnn":
            model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        elif arch == "retinanet":
            model.head.classification_head = RetinaNetClassificationHead(256, anchors.num_anchors_per_location()[0], 2,
                                                                         norm_layer=partial(nn.GroupNorm, 32))
        else:
            model.head.classification_head = FCOSClassificationHead(256, 1, 2)
        model.backbone = Bf16(model.backbone)
        return model, 3

    body_strides = (4, 8, 16, 32) if arch == "fasterrcnn" else (8, 16, 32)
    extra = LastLevelMaxPool() if arch == "fasterrcnn" else LastLevelP6P7(256, 256)
    timm_fpn = TimmFPN(backbone, body_strides, extra, pretrained, drop_path)
    cfg = timm_fpn.body.pretrained_cfg  # the 3 input channels are summed into 1, so use the channel means
    fpn = Bf16(timm_fpn)
    common |= dict(image_mean=[sum(cfg["mean"]) / 3], image_std=[sum(cfg["std"]) / 3])
    if arch == "fasterrcnn":
        return FasterRCNN(fpn, num_classes=2, rpn_anchor_generator=anchors, **common, **score), 1
    if arch == "retinanet":
        return RetinaNet(fpn, num_classes=2, anchor_generator=anchors, **common, **score), 1
    return FCOS(fpn, num_classes=2, anchor_generator=anchors, **common, **score), 1
