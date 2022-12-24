import torch
import numpy as np
from typing import Any, Dict, Tuple
from pytorchvideo.transforms.functional import convert_to_one_hot


def HorizontalFlip(batch_img):
    batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])
    return batch_img


def CenterCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = int(round((w - tw)) / 2.0)
    y1 = int(round((h - th)) / 2.0)
    img = batch_img[:, y1 : y1 + th, x1 : x1 + tw]
    return img


def RandomCrop(batch_img, size, movement_x):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = int(round((w - tw)) / 2.0) + movement_x
    y1 = int(round((h - th)) / 2.0)
    img = batch_img[:, y1 : y1 + th, x1 : x1 + tw]
    return img


def _mix_labels(
    labels: torch.Tensor,
    num_classes: int,
    lam: float = 1.0,
    label_smoothing: float = 0.0,
):
    """
    This function converts class indices to one-hot vectors and mix labels, given the
    number of classes.

    Args:
        labels (torch.Tensor): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixing labels.
        label_smoothing (float): Label smoothing value.
    """
    labels1 = convert_to_one_hot(labels, num_classes, label_smoothing)
    labels2 = convert_to_one_hot(labels.flip(0), num_classes, label_smoothing)
    return labels1 * lam + labels2 * (1.0 - lam)


class MixUpVideoImage(torch.nn.Module):
    """
    Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    ### Video Mixup References ###
    - https://github.com/facebookresearch/pytorchvideo/tree/main/pytorchvideo/transforms
    - Mixup Documentatin: https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/mix.html#MixUp

    ### Image Mixup References ###
    - https://discuss.pytorch.org/t/torchvision-example-on-mixup-and-crossentropyloss/142561
    - https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
    """

    def __init__(
        self,
        alpha: float = 0.4,
        label_smoothing: float = 0.0,
        num_classes: int = 500,
        one_hot: bool = True,
    ) -> None:
        """
        This implements MixUp for videos.
        Args:
            alpha (float): Mixup alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.mixup_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.one_hot = one_hot

    def forward(
        self,
        x_image: torch.Tensor,
        x_flipped_image: torch.Tensor,
        x_video: torch.Tensor,
        x_flipped_video: torch.Tensor,
        labels: torch.Tensor,
        **args: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.
        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
            Optional: x_audio: Audio input tensor.
        """
        assert x_video.size(0) > 1, "MixUp cannot be applied to a single instance."
        mixup_lambda = self.mixup_beta_sampler.sample()

        # in-batch mix images
        x_image_flipped = x_image.flip(0).mul_(1.0 - mixup_lambda)
        x_image.mul_(mixup_lambda).add_(x_image_flipped)
        x_flipped_image_flipped = x_flipped_image.flip(0).mul_(1.0 - mixup_lambda)
        x_flipped_image.mul_(mixup_lambda).add_(x_flipped_image_flipped)

        # in-batch mix videos
        x_video_flipped = x_video.flip(0).mul_(1.0 - mixup_lambda)
        x_video.mul_(mixup_lambda).add_(x_video_flipped)
        x_flipped_video_flipped = x_flipped_video.flip(0).mul_(1.0 - mixup_lambda)
        x_flipped_video.mul_(mixup_lambda).add_(x_flipped_video_flipped)

        # mix labels
        new_labels = _mix_labels(
            labels=labels,
            num_classes=self.num_classes,
            lam=mixup_lambda,
            label_smoothing=self.label_smoothing,
        )

        return x_image, x_flipped_image, x_video, x_flipped_video, new_labels


__all__ = ["HorizontalFlip", "CenterCrop", "RandomCrop", "MixUpVideoImage"]
