# RFdiffusion3 experiment matrix for 13 PDB targets

## Goals
- Measure whether lightly perturbed backbones can be recovered.
- Compare the effect of coordinate locking, hydrogen-bond constraints, partial diffusion, and symmetry constraints.
- Identify targets that are naturally recoverable versus targets that need strong conditioning.

## Target tiers
- Tier 1: small monomers with clean single-chain folds
- Tier 2: medium single-domain proteins with loops or modest oligomeric context
- Tier 3: oligomers, symmetry-dependent proteins, or structures with cofactors / complex interfaces

## Experimental factors
1. Backbone perturbation
   - `none`
   - `0.2A`
   - `0.5A`
   - `1.0A`

2. Structural conditioning
   - `unconditional`
   - `fixed_coords`
   - `fixed_coords + hbonds`
   - `fixed_coords + partial_diffusion`
   - `fixed_coords + hbonds + partial_diffusion`
   - `+ symmetry` when biologically justified

3. Sampling
   - 20 to 100 samples per condition
   - fixed random seeds for comparability

## Recommended baseline grid
For each target:
- 1 x unconditional
- 3 x perturbation levels under fixed coords
- 3 x perturbation levels under fixed coords + hbonds
- 3 x perturbation levels under fixed coords + partial diffusion
- 3 x perturbation levels under fixed coords + hbonds + partial diffusion

## Suggested success metrics
- Backbone RMSD
- TM-score
- Motif RMSD for fixed functional regions
- Contact-map similarity
- Secondary structure retention
- Clash score / chain break count

## Recommended reporting table
| target | chain | class | perturbation | constraints | samples | success_rate | best_rmsd | best_tm | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---|

## Practical order of operations
1. Run unconditional baseline on 2-3 pilot targets.
2. Run fixed-coordinates only.
3. Add hydrogen-bond constraints.
4. Add partial diffusion.
5. Add symmetry only for oligomeric or symmetric targets.
6. Expand to all 13 targets once the pipeline is stable.
