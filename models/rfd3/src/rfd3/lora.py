import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        self.merge_weights = merge_weights
        self.merged = False
        self.out_shape = getattr(base_layer, "out_shape", None)

        if r > 0:
            self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        self.freeze_base_layer()

    def freeze_base_layer(self):
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.base_layer(x)
        if self.r > 0 and not self.merged:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            out = out + lora_out.reshape_as(out)
        if self.out_shape is not None:
            return out.reshape(x.shape[:-1] + self.out_shape)
        return out

    def merge(self):
        if self.r <= 0 or self.merged:
            return
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
        self.merged = True

    def unmerge(self):
        if self.r <= 0 or not self.merged:
            return
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.base_layer.weight.data -= delta_w.to(self.base_layer.weight.dtype)
        self.merged = False


def _should_replace_linear(module_name: str, target_keywords: Iterable[str]) -> bool:
    return any(keyword in module_name for keyword in target_keywords)


def inject_lora_into_model(
    model: nn.Module,
    target_keywords: list[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    freeze_all: bool = True,
) -> nn.Module:
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False

    replaced_count = 0

    def _replace(parent: nn.Module, prefix: str = ""):
        nonlocal replaced_count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if _should_replace_linear(full_name, target_keywords):
                if isinstance(child, nn.Linear):
                    setattr(
                        parent,
                        child_name,
                        LoRALinear(
                            base_layer=child,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        ),
                    )
                    replaced_count += 1
                elif child.__class__.__name__ == "MultiDimLinear" and hasattr(child, "weight"):
                    setattr(
                        parent,
                        child_name,
                        LoRALinear(
                            base_layer=child,
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        ),
                    )
                    replaced_count += 1
                else:
                    _replace(child, full_name)
            else:
                _replace(child, full_name)

    _replace(model)

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True

    if replaced_count == 0:
        raise RuntimeError(
            "No modules were replaced by LoRA. Check target_keywords against the actual module names."
        )

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total