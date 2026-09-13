"""Validation lDDT metrics for generated structures."""

from __future__ import annotations

from foundry.metrics.metric import Metric
from rfd3.metrics.losses import smoothed_lddt_loss


class LDDTMetrics(Metric):
    """Compute the same differentiable lDDT definition used by RFD3 training.

    The loss helper returns ``1 - lDDT`` in its extra values, so this metric
    converts those values back to the conventional higher-is-better scale.
    """

    @property
    def kwargs_to_compute_args(self):
        return {
            "X_L": ("network_output", "X_L"),
            "X_gt_L": ("extra_info", "X_gt_L"),
            "crd_mask_L": ("extra_info", "crd_mask_L"),
            "is_dna": ("network_input", "f", "is_dna"),
            "is_rna": ("network_input", "f", "is_rna"),
            "tok_idx": ("network_input", "f", "atom_to_token_map"),
            "is_virtual": ("network_input", "f", "is_virtual"),
        }

    def compute(
        self,
        X_L,
        X_gt_L,
        crd_mask_L,
        is_dna,
        is_rna,
        tok_idx,
        is_virtual,
    ):
        _, extras = smoothed_lddt_loss(
            X_L=X_L,
            X_gt_L=X_gt_L,
            crd_mask_L=crd_mask_L,
            is_dna=is_dna,
            is_rna=is_rna,
            tok_idx=tok_idx,
            is_virtual=is_virtual,
            return_extras=True,
        )
        return {
            key: float(1.0 - value.item())
            for key, value in extras.items()
            if key in {"mean_lddt", "mean_lddt_protein"}
        }
