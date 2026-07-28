# RFdiffusion3 command templates

## Conventions
- `INPUT_PDB` refers to a prepared file in `rfdiffusion3_inputs/single_chain/` or `rfdiffusion3_inputs/backbone_only/`
- `OUT_DIR` is a run-specific output directory
- `FIXED_POSITIONS_JSONL`, `HBONDS_JSONL`, `MASK_JSONL` are target-specific files generated from residue plans

## Template 1: unconditional baseline
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50
```

## Template 2: fixed-coordinates run
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50 \
  scaffoldguided.inpaint_str=True \
  scaffoldguided.fixed_positions_jsonl=FIXED_POSITIONS_JSONL
```

## Template 3: fixed coordinates + hydrogen bonds
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50 \
  scaffoldguided.inpaint_str=True \
  scaffoldguided.fixed_positions_jsonl=FIXED_POSITIONS_JSONL \
  constraints.hbonds_jsonl=HBONDS_JSONL
```

## Template 4: partial diffusion
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50 \
  diffuser.partial_diffusion=True \
  diffuser.partial_diffusion_mask=MASK_JSONL
```

## Template 5: fixed coordinates + partial diffusion + hbonds
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50 \
  scaffoldguided.inpaint_str=True \
  scaffoldguided.fixed_positions_jsonl=FIXED_POSITIONS_JSONL \
  diffuser.partial_diffusion=True \
  diffuser.partial_diffusion_mask=MASK_JSONL \
  constraints.hbonds_jsonl=HBONDS_JSONL
```

## Template 6: symmetry-conditioned oligomers
```bash
python /path/to/RFdiffusion3/scripts/run_inference.py \
  inference.input_pdb=INPUT_PDB \
  inference.output_dir=OUT_DIR \
  inference.num_designs=50 \
  symmetry.enabled=True \
  symmetry.type=Cn
```

## Recommended run naming
- `target_baseline`
- `target_fixed`
- `target_fixed_hbonds`
- `target_partial`
- `target_fixed_partial_hbonds`
- `target_symmetry`
