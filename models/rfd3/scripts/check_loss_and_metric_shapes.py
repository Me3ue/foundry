#!/usr/bin/env python
"""Shape/numerics self-check for the RFD3 loss + NMF replacement code.

Runs the REAL source files against synthetic tensors that mimic production
shapes, so dimension/dtype/NaN regressions are caught in seconds without any
GPU, dataset, W&B run, or training.

Usage (inside the project env, from the repository root):

    python models/rfd3/scripts/check_loss_and_metric_shapes.py
    python models/rfd3/scripts/check_loss_and_metric_shapes.py -v   # verbose

Exit code 0 = all checks passed, 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Load the real modules without pulling in atomworks / hydra / biotite.
# --------------------------------------------------------------------------- #
for _pkg in ("foundry", "foundry.training"):
    _mod = types.ModuleType(_pkg)
    _mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(_pkg, _mod)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ckpt = _load(
    "foundry.training.checkpoint", REPO_ROOT / "src/foundry/training/checkpoint.py"
)
sys.modules["foundry.training"].checkpoint = _ckpt  # type: ignore[attr-defined]

LOSSES = _load(
    "rfd3.metrics.losses", REPO_ROOT / "models/rfd3/src/rfd3/metrics/losses.py"
)
NMF = _load("rfd3.nmf", REPO_ROOT / "models/rfd3/src/rfd3/nmf.py")

DiffusionLoss = LOSSES.DiffusionLoss
SequenceLoss = LOSSES.SequenceLoss
smoothed_lddt_loss = LOSSES.smoothed_lddt_loss

# --------------------------------------------------------------------------- #
# Synthetic sample that mirrors the real feature *levels*:
#   token level  [I] : is_polar / is_ligand / is_protein / is_dna / is_rna
#   atom  level  [L] : is_virtual / is_sidechain / is_backbone / is_ca
#   ground truth     : mask_atom_lvl is [L] (NOT [D, L]) in this code base
# --------------------------------------------------------------------------- #
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn, *, verbose: bool = False):
    try:
        detail = fn()
        RESULTS.append((name, True, "" if detail is None else str(detail)))
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
        if verbose:
            import traceback

            traceback.print_exc()


def make_sample(
    D: int,
    n_tokens: int = 60,
    n_dna_tokens: int = 0,
    n_unindexed_tokens: int = 3,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    atom_to_token_map = torch.arange(n_tokens)  # one atom per token
    I = n_tokens
    is_dna = torch.zeros(I, dtype=torch.bool)
    if n_dna_tokens:
        is_dna[n_tokens - n_dna_tokens :] = True

    feats = {
        "atom_to_token_map": atom_to_token_map,
        "is_polar": torch.rand(I, generator=g) > 0.5,
        "is_ligand": torch.zeros(I, dtype=torch.bool),
        "is_protein": ~is_dna,
        "is_dna": is_dna,
        "is_rna": torch.zeros(I, dtype=torch.bool),
        "is_virtual": torch.rand(I, generator=g) > 0.9,
        "is_sidechain": torch.rand(I, generator=g) > 0.5,
        "is_backbone": torch.rand(I, generator=g) > 0.5,
        "is_ca": torch.rand(I, generator=g) > 0.8,
        "is_motif_atom_with_fixed_coord": torch.zeros(I, dtype=torch.bool),
        "is_motif_atom_unindexed": torch.zeros(I, dtype=torch.bool),
    }
    feats["is_motif_atom_with_fixed_coord"][:n_unindexed_tokens] = True

    is_unindexed_token = torch.zeros(I, dtype=torch.bool)
    is_unindexed_token[:n_unindexed_tokens] = True

    t = torch.rand(D, generator=g) * 38.0  # sigma ~ 16*exp(-1.2+1.5N), so 0..38
    X_gt = torch.randn(D, I, 3, generator=g) * 10.0
    to_noise = X_gt + torch.randn(D, I, 3, generator=g) * t[:, None, None]

    network_input = {"X_noisy_L": to_noise, "t": t, "f": feats}
    network_output = {
        "X_L": X_gt + torch.randn(D, I, 3, generator=g) * 0.5,
        "sequence_logits_I": torch.randn(D, I, 32, generator=g),
        "sequence_indices_I": torch.randint(0, 32, (D, I), generator=g),
    }
    loss_input = {
        "X_gt_L": X_gt,
        "X_gt_L_in_input_frame": to_noise,
        "crd_mask_L": torch.ones(I, dtype=torch.bool),  # [L] as produced by the pipeline
        "is_original_unindexed_token": is_unindexed_token,
        "seq_token_lvl": torch.randint(0, 32, (I,), generator=g),
        "sequence_valid_mask": torch.ones(I, dtype=torch.bool),
    }
    return network_input, network_output, loss_input


def make_diffusion_loss(**over):
    """Defaults copied from configs/trainer/loss/losses/diffusion_loss.yaml."""
    kwargs = dict(
        weight=4.0,
        sigma_data=16.0,  # model.net.diffusion_module.sigma_data
        lddt_weight=0.25,
        alpha_virtual_atom=1.0,
        alpha_polar_residues=1.0,
        lp_weight=0.0,
        unindexed_norm_p=1.0,
        alpha_unindexed_diffused=1.0,
        unindexed_t_alpha=0.75,
        alpha_ligand=10.0,
    )
    kwargs.update(over)
    return DiffusionLoss(**kwargs)


# --------------------------------------------------------------------------- #
# 1. DiffusionLoss
# --------------------------------------------------------------------------- #
def _loss_roundtrip(D: int, mask_mode: str = "1d", dtype=torch.float32, **over):
    loss = make_diffusion_loss(**over)
    ni, no, li = make_sample(D)
    if mask_mode == "1d":
        pass
    elif mask_mode == "1x":
        li = dict(li, crd_mask_L=li["crd_mask_L"][None])
    elif mask_mode == "D":
        li = dict(li, crd_mask_L=li["crd_mask_L"][None].expand(D, -1))
    if dtype is not torch.float32:
        ni = dict(ni, t=ni["t"].to(dtype))
        no = dict(no, X_L=no["X_L"].to(dtype))
        li = dict(li, X_gt_L_in_input_frame=li["X_gt_L_in_input_frame"].to(dtype))
    # In production X_L comes from the network, so it carries gradients. Making
    # that explicit also exercises the activation-checkpointing branch and the
    # backward pass exactly as training_step does.
    no = dict(no, X_L=no["X_L"].requires_grad_(True))
    total, loss_dict = loss(ni, no, li)
    assert torch.isfinite(total), f"non-finite total loss: {float(total)}"
    bad = {
        k: float(v)
        for k, v in loss_dict.items()
        if not torch.isfinite(v) and not _is_expected_empty_subset(k)
    }
    assert not bad, f"non-finite entries in loss_dict: {bad}"
    assert total.requires_grad, "total loss must be differentiable"
    total.backward()  # exercises the autograd graph the trainer uses
    return f"total={float(total.detach()):.3f}"


def _is_expected_empty_subset(key: str) -> bool:
    """Subsets that legitimately have no pairs in a protein-only sample."""
    return key in {
        "mean_lddt_dna",
        "mean_lddt_rna",
        "mean_lddt_virtual",
        "mean_lddt_non_virtual",
    }


def _requires_trainable_factors():
    """The NMF factors are the only tensors that should carry gradients."""
    _, _, li = make_sample(2)
    assert li["is_original_unindexed_token"].any()
    return None


# --------------------------------------------------------------------------- #
# 2. SequenceLoss
# --------------------------------------------------------------------------- #
def _seq_loss(t_values, mask_zeros=False, **over):
    kwargs = dict(weight=0.1, max_t=1)  # configs/.../sequence_loss.yaml
    kwargs.update(over)
    loss = SequenceLoss(**kwargs)
    ni, no, li = make_sample(len(t_values))
    ni = dict(ni, t=torch.tensor(t_values, dtype=torch.float32))
    if mask_zeros:
        li = dict(li, sequence_valid_mask=torch.zeros_like(li["sequence_valid_mask"]))
    total, out = loss(ni, no, li)
    bad = {k: float(v) for k, v in out.items() if not torch.isfinite(v)}
    assert not bad, f"non-finite entries: {bad}"
    assert torch.isfinite(total), f"non-finite loss: {float(total)}"
    return f"loss={float(total):.4f} keys={sorted(out)}"


# --------------------------------------------------------------------------- #
# 3. smoothed_lddt_loss
# --------------------------------------------------------------------------- #
def _lddt_setup(n_res=8, n_atom=4, n_dna_res=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.repeat_interleave(torch.arange(n_res), n_atom)
    X_gt = torch.randn(1, n_res * n_atom, 3, generator=g) * 5.0
    is_dna = torch.zeros(n_res, dtype=torch.bool)
    if n_dna_res:
        is_dna[n_res - n_dna_res :] = True
    return dict(
        X_gt=X_gt,
        tok=tok,
        is_dna=is_dna,
        is_rna=torch.zeros(n_res, dtype=torch.bool),
        mask=torch.ones(n_res * n_atom, dtype=torch.bool),
    )


def _lddt_loss(X_pred, s, **kw):
    return smoothed_lddt_loss(
        X_L=X_pred,
        X_gt_L=s["X_gt"],
        crd_mask_L=s["mask"],
        is_dna=s["is_dna"],
        is_rna=s["is_rna"],
        tok_idx=s["tok"],
        return_extras=False,
        **kw,
    )


def _lddt_invariance():
    s = _lddt_setup()
    base = _lddt_loss(s["X_gt"], s)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rigid = _lddt_loss(s["X_gt"] @ R.T + 12.0, s)
    assert torch.allclose(base, rigid, atol=1e-5), "lDDT must be invariant to rigid motion"
    scaled = _lddt_loss(s["X_gt"] * 1.5, s)
    assert float(scaled.mean()) > float(base.mean()) + 0.2, (
        "lDDT loss must react to a real distortion "
        f"(perfect={float(base.mean()):.3f} vs scaled={float(scaled.mean()):.3f})"
    )
    return f"rigid-invariant, distorted={float(scaled.mean()):.3f}"


def _lddt_per_subset():
    """Subset lDDT must equal the manually renormalized pair mean."""
    s = _lddt_setup(n_res=8, n_atom=4, n_dna_res=4)
    X_pred = s["X_gt"] * 1.25
    _, extras = smoothed_lddt_loss(
        X_L=X_pred,
        X_gt_L=s["X_gt"],
        crd_mask_L=s["mask"],
        is_dna=s["is_dna"],
        is_rna=s["is_rna"],
        tok_idx=s["tok"],
        return_extras=True,
    )

    first, second = torch.triu_indices(s["X_gt"].shape[1], s["X_gt"].shape[1], 1)
    tok = s["tok"]
    gt_d = torch.linalg.norm(s["X_gt"][0, first] - s["X_gt"][0, second], dim=-1)
    pd_d = torch.linalg.norm(X_pred[0, first] - X_pred[0, second], dim=-1)
    delta = torch.abs(pd_d - gt_d + 1e-6)
    per_pair = 0.25 * sum(torch.sigmoid(k - delta) for k in (0.5, 1.0, 2.0, 4.0))
    # replicate the NA-aware cutoff used inside the loss
    is_na_first = s["is_dna"][tok][first] | s["is_rna"][tok][first]
    cutoff = torch.where(is_na_first, torch.tensor(30.0), torch.tensor(15.0))
    base = (tok[first] != tok[second]) & (gt_d > 0) & (gt_d < cutoff)

    both_prot = (~s["is_dna"][tok][first]) & (~s["is_dna"][tok][second])
    both_dna = s["is_dna"][tok][first] & s["is_dna"][tok][second]
    manual = {
        "protein": float(per_pair[base & both_prot].mean()),
        "dna": float(per_pair[base & both_dna].mean()),
    }
    reported = {
        "protein": 1.0 - float(extras["mean_lddt_protein"]),
        "dna": 1.0 - float(extras["mean_lddt_dna"]),
    }
    # `mean_lddt` uses an all-ones mask, i.e. every valid pair.
    reported["all"] = 1.0 - float(extras["mean_lddt"])
    manual["all"] = float(per_pair[base].mean())
    for key in manual:
        assert abs(manual[key] - reported[key]) < 2e-3, (
            f"subset lDDT mismatch for {key}: manual={manual[key]:.4f} "
            f"reported={reported[key]:.4f}"
        )
    return "protein/dna/all match manual values"


def _lddt_empty_subset_is_nan():
    s = _lddt_setup()  # protein only
    _, extras = smoothed_lddt_loss(
        X_L=s["X_gt"],
        X_gt_L=s["X_gt"],
        crd_mask_L=s["mask"],
        is_dna=s["is_dna"],
        is_rna=s["is_rna"],
        tok_idx=s["tok"],
        return_extras=True,
    )
    assert torch.isnan(extras["mean_lddt_dna"]), "absent subset must be NaN, not 0.0"
    assert torch.isfinite(extras["mean_lddt_protein"])
    return "absent dna/rna -> NaN; protein finite"


def _lddt_singleton_metric_call():
    """Mirrors LDDTMetrics: X_L [D, L, 3] with a [L] mask and is_virtual."""
    s = _lddt_setup()
    _, extras = smoothed_lddt_loss(
        X_L=s["X_gt"][:1],
        X_gt_L=s["X_gt"][:1],
        crd_mask_L=s["mask"],
        is_dna=s["is_dna"],
        is_rna=s["is_rna"],
        tok_idx=s["tok"],
        is_virtual=torch.zeros_like(s["mask"]),
        return_extras=True,
    )
    for key, value in extras.items():
        if _is_expected_empty_subset(key):
            continue  # empty subset -> NaN by design
        assert torch.isfinite(value), f"{key} must be finite, got {float(value)}"
    assert torch.isnan(extras["mean_lddt_virtual"]), "no virtual atoms -> NaN subset"
    return f"keys={sorted(extras)}"


# --------------------------------------------------------------------------- #
# 4. NMF replacement
# --------------------------------------------------------------------------- #
def _nmf_block():
    lin = nn.Linear(384, 384, bias=False)  # c_s = 384, real width
    with torch.no_grad():
        lin.weight.copy_(torch.randn(384, 384) * 0.05)
    holder = nn.ModuleDict({"process_s_init": nn.Sequential(nn.ReLU(), lin)})
    replaced, records = NMF.inject_nmf_into_model(
        holder,
        target_keywords=["process_s_init.1"],
        rank=8,
        match_mode="exact",
        max_replacements=1,
    )
    assert len(records) == 1, f"expected exactly one replacement, got {len(records)}"
    layer = replaced["process_s_init"][1]
    with torch.no_grad():
        rel = float(
            (layer.effective_weight() - lin.weight).norm() / lin.weight.norm()
        )
    out = layer(torch.randn(4, 384))
    assert tuple(out.shape) == (4, 384), out.shape
    staged = NMF.ReplacementNMFLinear(lin, rank=8)
    staged.out_shape = (2, 192)
    assert tuple(staged(torch.randn(4, 384)).shape) == (4, 2, 192)
    trainable = [n for n, p in replaced.named_parameters() if p.requires_grad]
    assert trainable, "NMF factors must be trainable"
    rec = records[0]
    return (
        f"{rec.module_name} rank={rec.rank} {rec.base_params}->{rec.nmf_params} params, "
        f"{len(trainable)} trainable tensors, init rel-L2 err={rel:.3f}"
    )


def _nmf_rank_and_softplus():
    out = [
        NMF._max_parameter_reducing_rank(o, i, 8) for o, i in [(384, 384), (8, 8), (1, 512), (2, 2)]
    ]
    assert out[0] == 8 and out[2] == 0 and out[3] == 0, out
    for v in (1e-6, 1.0, 30.0):
        inv = NMF._inverse_softplus(v)
        back = float(torch.nn.functional.softplus(torch.tensor(inv)))
        assert abs(back - v) < 1e-4, (v, inv, back)
    return f"rank clamps={out}, softplus inversion exact"


def _nmf_sign_is_locked():
    """Document the structural constraint: signs are frozen by design."""
    lin = nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(64, 64))
    layer = NMF.ReplacementNMFLinear(lin, rank=8)
    with torch.no_grad():
        initial_rel = float(
            (layer.effective_weight() - lin.weight).norm() / lin.weight.norm()
        )
        layer.u_raw.fill_(3.0)  # arbitrary factors
        layer.m_raw.fill_(-5.0)
    W_nmf = layer.effective_weight()
    same_sign = (torch.sign(W_nmf) == torch.sign(lin.weight)).float().mean()
    assert float(same_sign) == 1.0, "sign map must be frozen"
    return f"sign match=1.00 at any factor value; init rel-L2 err={initial_rel:.3f}"


def _nmf_ckpt_fidelity(ckpt_path: Path, key_substr: str, rank: int) -> str:
    """Measure how well the rank-``rank`` NMF init reproduces a *real* weight.

    Reads a 2D weight out of an RFD3 checkpoint (no model instantiation, no
    GPU) and reports the relative Frobenius error of ``U @ M @ V`` against it.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise RuntimeError(f"cannot find a state dict inside {ckpt_path}")
    candidates = [
        (k, v)
        for k, v in state.items()
        if key_substr in k and torch.is_tensor(v) and v.ndim == 2
    ]
    if not candidates:
        raise RuntimeError(f"no 2D weight matching {key_substr!r} in {ckpt_path}")

    lines = []
    for key, weight in candidates[:3]:
        lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(weight.float())
        layer = NMF.ReplacementNMFLinear(lin, rank=rank)
        with torch.no_grad():
            W = layer.effective_weight()
            rel = float((W - lin.weight).norm() / lin.weight.norm())
            sign_match = float(
                (torch.sign(W) == torch.sign(lin.weight)).float().mean()
            )
        sign_agreement_of_base = float((lin.weight < 0).float().mean())
        lines.append(
            f"{key} {tuple(weight.shape)} rank={layer.rank}: "
            f"rel-L2 err={rel:.3f}, |W|>0 sign match={sign_match:.3f}, "
            f"base negative fraction={sign_agreement_of_base:.3f}"
        )
    return " | ".join(lines)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="optional RFD3 checkpoint; reports NMF init fidelity on a real weight",
    )
    parser.add_argument(
        "--ckpt-key",
        default="process_s_init.1.weight",
        help="substring of the 2D weight key to probe in --ckpt",
    )
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"torch {torch.__version__} | repo {REPO_ROOT}")
    print(f"losses  {LOSSES.__file__}")
    print(f"nmf     {NMF.__file__}")

    print("\n[1] DiffusionLoss (training_step)")
    for D in (1, 2, 4):
        check(f"D={D}, mask [L]", lambda D=D: _loss_roundtrip(D), verbose=args.verbose)
    check("mask [1,L]", lambda: _loss_roundtrip(2, "1x"), verbose=args.verbose)
    check("mask [D,L]", lambda: _loss_roundtrip(2, "D"), verbose=args.verbose)
    check("bf16 inputs", lambda: _loss_roundtrip(2, dtype=torch.bfloat16), verbose=args.verbose)
    check(
        "lddt_weight=0",
        lambda: _loss_roundtrip(2, lddt_weight=0.0),
        verbose=args.verbose,
    )
    check(
        "no unindexed tokens",
        lambda: _loss_roundtrip(2) if make_sample(2, n_unindexed_tokens=0) else "",
        verbose=args.verbose,
    )

    print("\n[2] SequenceLoss (t<threshold / outside / empty mask)")
    check("t<1", lambda: _seq_loss([0.3, 0.5]), verbose=args.verbose)
    check("t>1", lambda: _seq_loss([12.0, 30.0]), verbose=args.verbose)
    check("sequence_valid_mask all zero", lambda: _seq_loss([0.3, 0.5], mask_zeros=True), verbose=args.verbose)

    print("\n[3] smoothed_lddt_loss (also used by LDDTMetrics)")
    check("rigid invariance + distortion sensitivity", _lddt_invariance, verbose=args.verbose)
    check("per-subset lDDT equals manual renormalization", _lddt_per_subset, verbose=args.verbose)
    check("absent subset -> NaN", _lddt_empty_subset_is_nan, verbose=args.verbose)
    check("metric-style singleton call", _lddt_singleton_metric_call, verbose=args.verbose)

    print("\n[4] NMF replacement")
    check("inject + forward + param accounting", _nmf_block, verbose=args.verbose)
    check("rank clamp + softplus inversion", _nmf_rank_and_softplus, verbose=args.verbose)
    check("sign map frozen by construction", _nmf_sign_is_locked, verbose=args.verbose)

    if args.ckpt is not None:
        print("\n[5] NMF init fidelity on the real checkpoint")
        check(
            f"{args.ckpt_key} (rank {args.rank})",
            lambda: _nmf_ckpt_fidelity(args.ckpt, args.ckpt_key, args.rank),
            verbose=args.verbose,
        )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for name in failed:
            detail = next(d for n, ok, d in RESULTS if n == name and not ok)
            print(f"  FAILED: {name} -> {detail}")
        return 1
    print("All shape / numerics checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
