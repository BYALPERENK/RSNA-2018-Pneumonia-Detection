"""Image-level pneumonia classifier: a pretrained timm backbone with a 2-class or 3-class output."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3-class labels; "Lung Opacity" is the pneumonia class, so its probability is the pneumonia score
CLASSES = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]


class Classifier(nn.Module):
    """Takes float (N, 1, H, W) images in [0, 1] and normalizes them the way the backbone was pretrained.

    n_classes=2 uses one output with a sigmoid (pneumonia or not); n_classes=3 uses a softmax over CLASSES.
    drop_path is ignored by backbones that do not support it (DenseNet).
    """

    def __init__(self, name: str, n_classes: int, pretrained: bool = True, drop_path: float = 0.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        assert n_classes in (2, 3)
        self.n_classes, self.label_smoothing = n_classes, label_smoothing
        kwargs = dict(pretrained=pretrained, in_chans=1, num_classes=1 if n_classes == 2 else 3)
        try:
            self.backbone = timm.create_model(name, drop_path_rate=drop_path, **kwargs)
        except TypeError:
            self.backbone = timm.create_model(name, **kwargs)
        cfg = self.backbone.pretrained_cfg  # the 3 input channels are summed into 1, so use the channel means
        self.register_buffer("mean", torch.tensor(sum(cfg["mean"]) / 3).view(1, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(sum(cfg["std"]) / 3).view(1, 1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """labels are 3-class indices into CLASSES; the 2-class model only sees whether they are Lung Opacity."""
        if self.n_classes == 2:
            target = (labels == 2).float() * (1 - self.label_smoothing) + self.label_smoothing / 2
            return F.binary_cross_entropy_with_logits(logits.squeeze(1).float(), target)
        return F.cross_entropy(logits.float(), labels, label_smoothing=self.label_smoothing)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, 1) pneumonia probability for the 2-class model, (N, 3) class probabilities for the 3-class model."""
        return torch.sigmoid(logits.float()) if self.n_classes == 2 else torch.softmax(logits.float(), dim=1)
