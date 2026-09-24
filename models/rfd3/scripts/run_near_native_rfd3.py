#!/usr/bin/env python3
"""Create reproducible near-native RFD3 partial-diffusion benchmarks.

This is a *local reconstruction* experiment, not evidence of de-novo fold
invention: low partial noise and fixed backbone anchors deliberately retain
information from the experimental structure.  It generates validated RFD3
InputSpecifications, a run manifest, and invokes RFD3 with trajectory output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_IDS = "3BTA 5N0B 1MDT 1DM0 2AAI 1ABR 4UY2 4JTA 1QLX 7UMQ 5OQV 5O3L 6A6B".split()
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def require_usable_nvidia_gpu() -> None:
    """Fail early with a clear driver/runtime diagnosis before loading RFD3."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError(
            "nvidia-smi is unavailable. RFD3 inference requires a CUDA-enabled NVIDIA GPU "
            "visible to this process (including inside containers)."
        )
    result = subprocess.run(
        [executable, "-L"], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "NVIDIA driver/NVML is unavailable, so RFD3 cannot allocate GPU memory. "
            "This is a system CUDA driver/container passthrough problem, not an input "
            f"constraint problem. nvidia-smi reported: {detail}"
        )


def sha256(path: Path) -> str:
    """Return a stable content hash for the exact input coordinate file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def residue_key(chain: str, number: str, insertion_code: str = "") -> str:
    """Return the RFD3 selector used by InputSelection.

    RFD3's dialect-2 selector lookup is ``chain_id + res_id`` and does not
    include PDB insertion codes.  HBPLUS represents a blank insertion code as
    ``-``; appending it (e.g. ``A1548-``) makes an invalid RFD3 component.
    """
    return f"{chain}{number}"


def inspect_rfd3_input(path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Read the same biological assembly and residue IDs that RFD3 will use.

    Downloaded/deposited PDB records are not necessarily identical to the
    AtomWorks assembly selected by RFD3: non-polymer records can be moved to a
    different chain and records can be excluded.  All constraint selectors
    must therefore be derived from RFD3's parsed AtomArray, never from raw PDB
    line numbers.
    """
    try:
        from rfd3.utils.inference import inference_load_
    except ImportError as error:
        raise RuntimeError("Run this script in the RFD3/Foundry environment.") from error
    array = inference_load_(path)["atom_array"]
    residues, atom_names = [], {}
    starts = []
    previous = None
    for index, (chain, residue_id, residue_name) in enumerate(
        zip(array.chain_id, array.res_id, array.res_name)
    ):
        key = (str(chain), int(residue_id), str(residue_name).upper())
        if key != previous:
            starts.append(index)
            previous = key
    starts.append(array.array_length())
    for start, stop in zip(starts[:-1], starts[1:]):
        token = array[start:stop]
        resname = str(token.res_name[0]).upper()
        if resname not in STANDARD_AA or "CA" not in token.atom_name:
            continue
        selector = f"{token.chain_id[0]}{int(token.res_id[0])}"
        if selector in atom_names:
            raise ValueError(f"{path}: RFD3 input has ambiguous selector {selector}")
        residues.append(selector)
        atom_names[selector] = set(map(str, token.atom_name))
    if len(residues) < 2:
        raise ValueError(f"{path}: RFD3 parsed assembly has fewer than two standard protein residues")
    return residues, atom_names


def fetch_pdb(pdb_id: str, structures_dir: Path) -> Path:
    """Fetch the deposited coordinate file; RFD3 builds biological assembly 1."""
    destination = structures_dir / f"{pdb_id.lower()}.pdb"
    if not destination.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {pdb_id}: {error}") from error
    return destination.resolve()


def native_hbond_selections(pdb_path: Path, hbplus: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return only donor/acceptor atoms in *geometrically detected* native H bonds.

    HBPLUS's .hb2 records are used instead of declaring every chemically
    possible donor/acceptor as active.  RFD3 conditions atom identities, not
    donor--acceptor pairing; the exact pair geometry remains an evaluation
    criterion rather than a hard RFD3 restraint.
    """
    hbplus_path = shutil.which(hbplus) or hbplus
    if not Path(hbplus_path).exists():
        raise ValueError(f"HBPLUS executable not found: {hbplus}")
    workdir = pdb_path.parent / f".{pdb_path.stem}_hbplus"
    workdir.mkdir(exist_ok=True)
    local_pdb = workdir / pdb_path.name
    shutil.copy2(pdb_path, local_pdb)
    try:
        subprocess.run([hbplus_path, "-h", "3.0", "-d", "3.5", str(local_pdb), str(local_pdb)],
                       cwd=workdir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        hb2 = local_pdb.with_suffix(".hb2")
        if not hb2.exists():
            raise RuntimeError("HBPLUS completed but did not create an .hb2 file")
        donors: dict[str, set[str]] = defaultdict(set)
        acceptors: dict[str, set[str]] = defaultdict(set)
        for line in hb2.read_text(errors="replace").splitlines()[8:]:
            if len(line) < 32:
                continue
            # Columns follow RFD3's hbonds_hbplus.py parser.
            donor = residue_key(line[0].strip() or "A", str(int(line[1:5].strip())), line[5])
            acceptor = residue_key(line[14].strip() or "A", str(int(line[15:19].strip())), line[19])
            donors[donor].add(line[9:13].strip())
            acceptors[acceptor].add(line[23:27].strip())
        return ({key: ",".join(sorted(value)) for key, value in donors.items()},
                {key: ",".join(sorted(value)) for key, value in acceptors.items()})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def make_spec(pdb_path: Path, anchor_stride: int | None, partial_t: float, sequence_mode: str,
              hbplus: str | None,
              symmetry: str | None) -> tuple[dict, dict]:
    residues, available_atoms = inspect_rfd3_input(pdb_path)
    anchors = residues[::anchor_stride] if anchor_stride is not None else []
    if anchor_stride is not None and residues[-1] not in anchors:
        anchors.append(residues[-1])
    spec: dict = {
        "input": str(pdb_path), "partial_t": partial_t,
        # No --anchor-stride means a no-fixed-coordinate ablation.
        "select_fixed_atoms": {residue: "BKBN" for residue in anchors},
        # This field selects residues *to unfix*: False fixes every input residue,
        # while True allows all input residue identities to be predicted.
        "select_unfixed_sequence": sequence_mode == "diffuse",
        "cif_parser_args": {"add_missing_atoms": False, "hydrogen_policy": "remove"},
        "extra": {"benchmark_type": "near_native_partial_diffusion", "input_sha256": sha256(pdb_path)},
    }
    hbond_info = {"mode": "off", "donors": 0, "acceptors": 0}
    if hbplus:
        donors, acceptors = native_hbond_selections(pdb_path, hbplus)
        # HBPLUS sees deposited records, whereas RFD3 conditions AtomWorks'
        # parsed biological assembly. Drop nonmatching records and atom names.
        donors = {
            residue: ",".join(atom for atom in names.split(",") if atom in available_atoms[residue])
            for residue, names in donors.items() if residue in available_atoms
        }
        acceptors = {
            residue: ",".join(atom for atom in names.split(",") if atom in available_atoms[residue])
            for residue, names in acceptors.items() if residue in available_atoms
        }
        donors = {residue: names for residue, names in donors.items() if names}
        acceptors = {residue: names for residue, names in acceptors.items() if names}
        if donors:
            spec["select_hbond_donor"] = donors
        if acceptors:
            spec["select_hbond_acceptor"] = acceptors
        hbond_info = {"mode": "native_hbplus", "donors": sum(len(x.split(",")) for x in donors.values()),
                      "acceptors": sum(len(x.split(",")) for x in acceptors.values())}
    if symmetry:
        spec["symmetry"] = {"id": symmetry, "is_symmetric_motif": True}
    return spec, {"residues": len(residues), "anchors": len(anchors), "hbond_conditioning": hbond_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, default=Path("inputs/pdb"))
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--partial-t", type=float, default=1.0, help="Noise in Å; <=2 Å is a near-native test.")
    parser.add_argument("--anchor-stride", type=int, default=5,
                        help="Fix N/CA/C/O every N residues; use 0 for no fixed-coordinate anchor ablation.")
    parser.add_argument("--sequence-mode", choices=("fixed", "diffuse"), default="fixed",
                        help="'fixed' preserves input residue identities; 'diffuse' lets RFD3 predict them.")
    parser.add_argument("--max-protein-residues", type=int, default=None,
                        help="Skip targets whose RFD3-parsed protein residue count exceeds this limit.")
    parser.add_argument("--hbplus", metavar="PATH", help="Enable native geometry-derived H-bond atom conditioning.")
    parser.add_argument("--symmetry", metavar="PDB_ID=C3", action="append", default=[],
                        help="Verified Cn/Dn symmetry; only use a pre-symmetrized input/assembly.")
    parser.add_argument("--designs-per-target", type=int, default=1,
                        help="Samples per target; use 8 only after a successful single-sample smoke test.")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20250308)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to an RFD3 .ckpt file; defaults to the registered 'rfd3' checkpoint.")
    parser.add_argument("--rfd3-command", default="rfd3 design")
    parser.add_argument("--low-memory-mode", action="store_true",
                        help="Use RFD3 chunked pairwise inference to lower GPU memory at the cost of speed.")
    parser.add_argument("--skip-rfd3-prevalidation", action="store_true",
                        help="Do not locally call DesignInputSpecification.safe_init() before launch.")
    parser.add_argument("--skip-gpu-preflight", action="store_true",
                        help="Skip nvidia-smi check (only for advanced/custom runtimes).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.partial_t <= 15 or args.anchor_stride < 0 or args.designs_per_target < 1 or args.timesteps < 2:
        parser.error("partial-t must be in (0,15], anchor-stride must be >=0, designs/timesteps must be positive (timesteps >=2)")
    anchor_stride = args.anchor_stride or None
    if args.max_protein_residues is not None and args.max_protein_residues < 2:
        parser.error("--max-protein-residues must be at least 2")

    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"Checkpoint does not exist or is not a file: {args.checkpoint}")

    symmetries = {}
    for item in args.symmetry:
        try:
            pdb_id, group = item.upper().split("=", 1)
        except ValueError:
            parser.error(f"Invalid --symmetry {item!r}; expected PDB_ID=C3 or PDB_ID=D2")
        if not (group.startswith(("C", "D")) and group[1:].isdigit()):
            parser.error("RFD3 supports only Cn and Dn symmetry")
        symmetries[pdb_id] = group
    if symmetries and any(pdb_id.upper() not in symmetries for pdb_id in args.ids):
        parser.error("Symmetric and asymmetric targets must be separate invocations.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.structures_dir.mkdir(parents=True, exist_ok=True)
    specs, targets, skipped = {}, {}, {}
    for raw_id in args.ids:
        pdb_id, path = raw_id.upper(), fetch_pdb(raw_id.upper(), args.structures_dir)
        residues, _ = inspect_rfd3_input(path)
        if args.max_protein_residues is not None and len(residues) > args.max_protein_residues:
            skipped[pdb_id] = {
                "reason": "protein_residue_limit",
                "protein_residues": len(residues),
                "max_protein_residues": args.max_protein_residues,
                "input": str(path),
            }
            print(f"Skipping {pdb_id}: {len(residues)} protein residues > limit {args.max_protein_residues}")
            continue
        spec, target = make_spec(path, anchor_stride, args.partial_t, args.sequence_mode,
                                 args.hbplus, symmetries.get(pdb_id))
        specs[f"{pdb_id}_near_native"] = spec
        targets[pdb_id] = {"input": str(path), "sha256": sha256(path), **target}
    if not specs:
        raise RuntimeError("No targets remain after applying --max-protein-residues.")

    input_json = args.out_dir / "near_native_inputs.json"
    input_json.write_text(json.dumps(specs, indent=2) + "\n")
    if not args.skip_rfd3_prevalidation:
        from rfd3.inference.input_parsing import DesignInputSpecification
        for example_id, spec in specs.items():
            try:
                DesignInputSpecification.safe_init(**spec)
            except Exception as error:
                raise RuntimeError(
                    f"Local RFD3 prevalidation failed for {example_id}; no inference was started."
                ) from error
    command = args.rfd3_command.split() + [
        f"out_dir={args.out_dir.resolve()}", f"inputs={input_json.resolve()}",
        f"diffusion_batch_size={args.designs_per_target}", "n_batches=1", "dump_trajectories=True",
        "prevalidate_inputs=True", "skip_existing=False", f"seed={args.seed}",
        f"inference_sampler.num_timesteps={args.timesteps}", "inference_sampler.allow_realignment=False",
        "align_trajectory_structures=True", f"low_memory_mode={args.low_memory_mode}",
    ]
    if args.checkpoint is not None:
        command.append(f"ckpt_path={args.checkpoint}")
    if symmetries:
        command.append("inference_sampler.kind=symmetry")
    parameters = vars(args).copy()
    parameters["out_dir"] = str(args.out_dir)
    parameters["structures_dir"] = str(args.structures_dir)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "targets": targets,
                "skipped_targets": skipped, "parameters": parameters, "command": command,
                "limitations": ["Partial diffusion is conditional reconstruction, not de-novo generation.",
                                "RFD3 H-bond conditioning is atom-wise, not a hard pairwise geometry restraint.",
                                "RFD3 currently loads biological assembly 1; verify it is the relevant experimental assembly."]}
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print("Wrote", input_json, "and", args.out_dir / "run_manifest.json")
    print("Running:", " ".join(command))
    if not args.dry_run:
        if not args.skip_gpu_preflight:
            require_usable_nvidia_gpu()
        subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
