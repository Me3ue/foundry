"""Validation-path unresolved-coordinate handling.

Experimental PDB holdout rows almost always contain unresolved atoms. The
training route fills those coordinates after cropping; validation used to skip
that fill, so ``X_noisy_L`` was NaN for nearly every holdout example.

The fill after ``CopyAnnotation`` must be wrapped in ``InferenceRoute``.
atomworks forbids applying ``PlaceUnresolvedToken*`` twice, and training
already ran those transforms on ``coord``.
"""
from __future__ import annotations

from rfd3.trainer.rfd3 import _is_cuda_oom, _skipped_validation_result
from rfd3.transforms.pipelines import get_crop_transform, get_diffusion_transforms


def test_diffusion_transforms_fill_unresolved_coords_only_at_inference():
    transforms = get_diffusion_transforms(sigma_data=16.0, diffusion_batch_size=1)
    names = [type(t).__name__ for t in transforms]
    assert "CopyAnnotation" in names
    copy_idx = names.index("CopyAnnotation")
    fill_atoms_route = transforms[copy_idx + 1]
    fill_seq_route = transforms[copy_idx + 2]
    assert type(fill_atoms_route).__name__ == "ConditionalRoute"
    assert type(fill_seq_route).__name__ == "ConditionalRoute"
    fill_atoms = fill_atoms_route.transform_map[True]
    fill_seq = fill_seq_route.transform_map[True]
    assert type(fill_atoms).__name__ == "PlaceUnresolvedTokenAtomsOnRepresentativeAtom"
    assert type(fill_seq).__name__ == "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence"
    assert fill_atoms.annotation_to_update == "coord_to_be_noised"
    assert fill_seq.annotation_to_update == "coord_to_be_noised"
    assert fill_seq.annotation_to_copy == "coord_to_be_noised"
    # Training must not re-run the fill (Identity on the False/training branch).
    assert type(fill_atoms_route.transform_map[False]).__name__ == "Identity"
    assert type(fill_seq_route.transform_map[False]).__name__ == "Identity"


def _crop_kwargs(**overrides):
    args = dict(
        crop_size=256,
        crop_center_cutoff_distance=15.0,
        crop_contiguous_probability=0.0,
        crop_spatial_probability=1.0,
        dna_contact_crop_probability=0.0,
        keep_full_binder_in_spatial_crop=False,
        max_binder_length=170,
        max_atoms_in_crop=1920,
        allowed_types="ALL",
    )
    args.update(overrides)
    return args


def test_holdout_max_atoms_enables_validation_cropping():
    crop = get_crop_transform(**_crop_kwargs(max_atoms_in_crop=1920))
    # SubsampleToTypes, RandomRoute(crop), TrainingRoute(fill), TrainingRoute(fill)
    assert type(crop[1]).__name__ == "RandomRoute"


def test_unbounded_inference_still_skips_cropping():
    crop = get_crop_transform(**_crop_kwargs(max_atoms_in_crop=None))
    assert type(crop[1]).__name__ == "ConditionalRoute"
    assert type(crop[1].transform_map[True]).__name__ == "Identity"
    assert type(crop[1].transform_map[False]).__name__ == "RandomRoute"


def test_cuda_oom_is_detected_and_skipped():
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 43.20 GiB"))
    assert not _is_cuda_oom(RuntimeError("shape mismatch"))
    skipped = _skipped_validation_result()
    assert skipped["skip"] is True
    assert skipped["metrics_output"] is None


def test_missing_specification_is_treated_as_empty_metadata():
    from rfd3.trainer.rfd3 import _get_example_specification

    assert _get_example_specification({}) == {}
    assert _get_example_specification({"specification": None}) == {}


def test_specification_metadata_is_preserved():
    from rfd3.trainer.rfd3 import _get_example_specification

    specification = {"example": "pdb_holdout"}
    assert _get_example_specification({"specification": specification}) is specification
