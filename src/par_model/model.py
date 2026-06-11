"""Multi-task ConvNeXt architecture used by the trained checkpoint."""

from collections.abc import Mapping

import timm
import torch
from torch import nn


class MultiTaskPAR(nn.Module):
    """Shared image backbone with one classification head per attribute."""

    def __init__(
        self,
        num_classes: Mapping[str, int],
        model_name: str = "convnext_small",
        *,
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(self.backbone.num_features, class_count)
                for task, class_count in num_classes.items()
            }
        )
        self.log_vars = nn.Parameter(torch.zeros(len(num_classes)))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.drop(self.backbone(images))
        return {task: head(features) for task, head in self.heads.items()}
