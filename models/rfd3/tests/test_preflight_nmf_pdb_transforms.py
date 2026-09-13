"""CPU tests for holdout preflight NaN screening helpers.

The previous preflight only treated transform exceptions and atomworks
fallback rows as failures. Validation still crashed on 9hql because
``X_noisy_L`` contained unresolved-coordinate NaNs after a successful
transform. These tests lock that check in without loading PDB files.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_preflight():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_nmf_pdb_transforms.py"
    spec = importlib.util.spec_from_file_location("preflight_nmf_pdb_transforms", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()


def _example(*, coords, noise=None, feats=None, example_id="{['pdb', 'interfaces']}{9hql}{1}{['A_1', 'B_1']}"):
    if noise is None:
        noise = torch.zeros_like(coords)
    return {
        "example_id": example_id,
        "coord_atom_lvl_to_be_noised": coords,
        "noise": noise,
        "feats": feats if feats is not None else {"ref_pos": torch.zeros(2, 3)},
    }


def test_clean_example_passes_validation_nan_check():
    preflight._check_validation_network_inputs(
        _example(coords=torch.zeros(4, 3))
    )


def test_x_noisy_nan_is_rejected_with_trainer_wording():
    coords = torch.zeros(4, 3)
    coords[1, 0] = float("nan")
    with pytest.raises(AssertionError, match=r"network_input \(X_noisy_L\).*Tensor contains NaNs!"):
        preflight._check_validation_network_inputs(_example(coords=coords))


def test_noise_nan_is_rejected():
    coords = torch.zeros(4, 3)
    noise = torch.zeros(4, 3)
    noise[0, 2] = float("nan")
    with pytest.raises(AssertionError, match="Tensor contains NaNs!"):
        preflight._check_validation_network_inputs(_example(coords=coords, noise=noise))


def test_feat_nans_are_rejected():
    with pytest.raises(AssertionError, match=r"NaN detected in `feats`"):
        preflight._check_validation_network_inputs(
            _example(
                coords=torch.zeros(2, 3),
                feats={"ref_pos": torch.tensor([[0.0, 1.0, float("nan")]])},
            )
        )


def test_numpy_coord_nans_are_rejected():
    coords = np.zeros((3, 3), dtype=np.float32)
    coords[2, 1] = np.nan
    with pytest.raises(AssertionError, match="contains NaNs"):
        preflight._check_validation_network_inputs(
            _example(coords=coords, noise=np.zeros_like(coords))
        )


def test_wrapped_batch_list_is_unwrapped():
    example = _example(coords=torch.zeros(2, 3))
    preflight._check_validation_network_inputs(preflight._as_example_dict([example]))


def test_error_group_for_x_noisy_nans():
    assert preflight._error_group(
        "AssertionError: network_input (X_noisy_L) for example_id: 9hql: Tensor contains NaNs!"
    ) == "network_input_nans"


def test_write_filter_unions_existing_ids(tmp_path):
    path = tmp_path / "pdb_holdout.yaml"
    path.write_text(
        '      - "pdb_id not in [\'7puh\', \'9hq9\']"\n',
        encoding="utf-8",
    )
    preflight._write_filter(path, {"9hql", "9hq9"})
    text = path.read_text(encoding="utf-8")
    assert "9hql" in text
    assert "7puh" in text
    assert "9hq9" in text
