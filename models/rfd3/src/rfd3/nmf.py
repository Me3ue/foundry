from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        return 0.0
    # log(exp(x) - 1) is stable enough for the small positive scales used here.
    return math.log(math.exp(value) - 1.0)


@dataclass
class NMFReplacementRecord:
    module_name: str
    module_type: str
    in_features: int
    out_features: int
    rank: int
    base_params: int
    nmf_params: int

    @property
    def reduction(self) -> int:
        return self.base_params - self.nmf_params


class ReplacementNMFLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        nmf_alpha: float = 1.0,
        nmf_eps: float = 1e-6,
    ):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError(f"ReplacementNMFLinear requires a layer with weight, got {type(base_layer)}")

        weight = base_layer.weight
        if weight.ndim != 2:
            raise TypeError(
                f"ReplacementNMFLinear only supports 2D weights, got shape {tuple(weight.shape)}"
            )

        self.base_layer = None
        self.rank = rank
        self.nmf_alpha = nmf_alpha
        self.nmf_eps = nmf_eps
        self.out_shape = getattr(base_layer, "out_shape", None)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        # Factorize the magnitude and preserve the pretrained signed map.
        self.register_buffer(
            "weight_sign",
            torch.where(weight.detach() < 0, -torch.ones_like(weight), torch.ones_like(weight)),
            persistent=True,
        )

        # Non-negative factors are parameterized through softplus to keep them strictly >= 0.
        # The factors approximate the base weight as W ~= U @ M @ V, with square core M.
        self.u_raw = nn.Parameter(torch.empty(self.out_features, rank))
        self.m_raw = nn.Parameter(torch.empty(rank, rank))
        self.v_raw = nn.Parameter(torch.empty(rank, self.in_features))

        bias = getattr(base_layer, "bias", None)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(weight.detach())

    def reset_parameters(self, base_weight: torch.Tensor | None = None) -> None:
        if base_weight is not None:
            target = base_weight.detach().abs().float().clamp_min(self.nmf_eps)
            generator = torch.Generator(device=target.device)
            generator.manual_seed(0)
            u = torch.rand(self.out_features, self.rank, device=target.device, dtype=target.dtype, generator=generator).clamp_min(self.nmf_eps)
            v = torch.rand(self.rank, self.in_features, device=target.device, dtype=target.dtype, generator=generator).clamp_min(self.nmf_eps)
            scale = target.mean().sqrt()
            u.mul_(scale); v.mul_(scale)
            for _ in range(25):
                v.mul_((u.transpose(0, 1) @ target) / (u.transpose(0, 1) @ u @ v).clamp_min(self.nmf_eps))
                u.mul_((target @ v.transpose(0, 1)) / (u @ (v @ v.transpose(0, 1))).clamp_min(self.nmf_eps))
                u.clamp_(min=self.nmf_eps); v.clamp_(min=self.nmf_eps)
            inv = lambda x: torch.where(x > 20, x, torch.log(torch.expm1(x).clamp_min(self.nmf_eps)))
            self.u_raw.data.copy_(inv(u).to(dtype=self.u_raw.dtype))
            self.v_raw.data.copy_(inv(v).to(dtype=self.v_raw.dtype))
            nn.init.constant_(self.m_raw, -10.0)
            self.m_raw.data.fill_diagonal_(_inverse_softplus(1.0))
            return
        nn.init.normal_(self.u_raw, mean=0.0, std=0.02)
        nn.init.normal_(self.m_raw, mean=0.0, std=0.02)
        nn.init.normal_(self.v_raw, mean=0.0, std=0.02)

    def _u(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.u_raw) + self.nmf_eps

    def _m(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.m_raw) + self.nmf_eps

    def _v(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.v_raw) + self.nmf_eps

    def effective_weight(self) -> torch.Tensor:
        magnitude = self._u() @ self._m() @ self._v()
        return self.nmf_alpha * self.weight_sign * magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.effective_weight()
        out = torch.nn.functional.linear(x, weight, self.bias)
        if self.out_shape is not None:
            return out.reshape(x.shape[:-1] + self.out_shape)
        return out


def _should_replace_linear(module_name: str, target_keywords: Iterable[str]) -> bool:
    return any(keyword in module_name for keyword in target_keywords)


def _max_parameter_reducing_rank(out_features: int, in_features: int, requested_rank: int) -> int:
    for candidate in range(min(requested_rank, out_features, in_features), 0, -1):
        if candidate * (out_features + in_features + candidate) < out_features * in_features:
            return candidate
    return 0


def _count_layer_params(base_layer: nn.Module, rank: int) -> tuple[int, int]:
    weight = getattr(base_layer, "weight", None)
    if weight is None or weight.ndim != 2:
        return 0, 0
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    base_params = out_features * in_features
    nmf_params = out_features * rank + rank * rank + rank * in_features
    bias = getattr(base_layer, "bias", None)
    if bias is not None:
        base_params += int(bias.numel())
        nmf_params += int(bias.numel())
    return base_params, nmf_params


def _is_replaceable_linear(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "ndim", 0) == 2


def inject_nmf_into_model(
    model: nn.Module,
    target_keywords: list[str],
    rank: int = 8,
    nmf_alpha: float = 1.0,
    nmf_eps: float = 1e-6,
    freeze_all: bool = True,
) -> tuple[nn.Module, list[NMFReplacementRecord]]:
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False

    replaced_count = 0
    records: list[NMFReplacementRecord] = []

    def _replace(parent: nn.Module, prefix: str = ""):
        nonlocal replaced_count
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if _should_replace_linear(full_name, target_keywords):
                if _is_replaceable_linear(child):
                    weight = getattr(child, "weight", None)
                    assert weight is not None
                    layer_rank = _max_parameter_reducing_rank(
                        int(weight.shape[0]), int(weight.shape[1]), rank
                    )
                    if layer_rank == 0:
                        continue
                    base_params, nmf_params = _count_layer_params(child, layer_rank)
                    setattr(
                        parent,
                        child_name,
                        ReplacementNMFLinear(
                            base_layer=child,
                            rank=layer_rank,
                            nmf_alpha=nmf_alpha,
                            nmf_eps=nmf_eps,
                        ),
                    )
                    replaced_count += 1
                    weight = getattr(child, "weight", None)
                    in_features = int(weight.shape[1]) if weight is not None and weight.ndim == 2 else 0
                    out_features = int(weight.shape[0]) if weight is not None and weight.ndim == 2 else 0
                    records.append(
                        NMFReplacementRecord(
                            module_name=full_name,
                            module_type=child.__class__.__name__,
                            in_features=in_features,
                            out_features=out_features,
                            rank=layer_rank,
                            base_params=base_params,
                            nmf_params=nmf_params,
                        )
                    )
                else:
                    _replace(child, full_name)
            else:
                _replace(child, full_name)

    _replace(model)

    for name, p in model.named_parameters():
        if any(token in name for token in ("u_raw", "m_raw", "v_raw")):
            p.requires_grad = True

    if replaced_count == 0:
        raise RuntimeError(
            "No modules were replaced by NMF. Check target_keywords against the actual module names."
        )

    return model, records


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
