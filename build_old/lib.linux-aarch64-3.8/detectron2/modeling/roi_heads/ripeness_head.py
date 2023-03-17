import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

ROI_RIPENESS_HEAD_REGISTRY = Registry("ROI_RIPENESS_HEAD")
ROI_RIPENESS_HEAD_REGISTRY.__doc__ = """
Registry for ripeness heads, which predicts instance ripeness given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

ROI_RIPENESS_LOSS_REGISTRY = Registry("ROI_RIPENESS_LOSS")
ROI_RIPENESS_LOSS_REGISTRY.__doc__ = """
Registry for ripeness heads, which calculates loss.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def initialize_module_params(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

def build_roi_ripeness_head(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_RIPENESS_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_RIPENESS_HEAD_REGISTRY.register()
class RipenessV1ConvXHead(nn.Module):
    """
    Fully convolutional DensePose head.
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize DensePose fully convolutional head

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        super(RipenessV1ConvXHead, self).__init__()
        # fmt: off
        hidden_dim           = cfg.MODEL.ROI_RIPENESS_HEAD.CONV_HEAD_DIM
        kernel_size          = cfg.MODEL.ROI_RIPENESS_HEAD.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.ROI_RIPENESS_HEAD.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

    def forward(self, features: torch.Tensor):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name
