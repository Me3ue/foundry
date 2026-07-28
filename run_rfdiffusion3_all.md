# Batch RFdiffusion3 runner

Use `run_rfdiffusion3_all.sh` after generating design inputs.

## Environment variables
- `RFD3_BIN` — path to the `rfd3` executable if it is not on your `PATH`
- `OUT_ROOT` — root output directory, default `logs/inference_outs`

## Usage
```bash
bash run_pipeline.sh
bash run_rfdiffusion3_all.sh
```

## What it does
For every JSON file in `rfdiffusion3_inputs/design_specs/`, it runs:
```bash
rfd3 design out_dir=<OUT_ROOT>/<PDB_ID>/0 inputs=<JSON> skip_existing=False dump_trajectories=True prevalidate_inputs=True
```
