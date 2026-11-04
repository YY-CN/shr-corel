# SHR-CoRel

SHR-CoRel predicts deception from a query response and one known truthful response from the same participant. A frozen semantic encoder supplies the prepared response embeddings. A query predictor is trained first, then frozen while a shared relation scorer learns the correction

`logit = query_logit + F(query, reference) - F(query, training_truthful_center)`.

The center uses truthful responses from the current training participants. The implementation supports participant-disjoint nested evaluation, participant-balanced fitting, and participant-macro reporting.

## Installation

The original experiment used Python 3.10, PyTorch 2.1.2 and NumPy 1.26.3. Install from this checkout using Python 3.10 and the pinned requirements. The default device is CPU. For CUDA, install the corresponding PyTorch 2.1.2 build suitable for your machine.

```bash
python -m venv .venv
# Activate this environment using your operating system's activation command.
python -m pip install -e .
python -m rel01 --help
python -m rel01 self-check
```

The self-check command uses the tests in the source checkout. The tests exercise model calculations, training weights, reference sampling, statistics, checkpoint recovery and restart consistency with synthetic inputs.

## Prepare the input features

Obtain [DOLOS](https://rose1.ntu.edu.sg/dataset/DOLOS/) and [MU3D](https://sc.lib.miamioh.edu/items/79ac38da-cb8e-4eff-be92-ee8765a5c72c) under their providers' access terms. The study uses 599 DOLOS responses from 41 eligible participants and 320 MU3D responses from 80 participants. Eligibility requires at least two truthful responses and at least one deceptive response per participant.

Prepare a float32 NPY matrix with shape `[599, 768]` for DOLOS or `[320, 768]` for MU3D. Rows must follow `embedding_row` in the corresponding `rel01/protocol/*_samples.csv`. The CSV supplies the original response identifiers, frozen participant assignments, labels, lexical token counts used for reference-length matching, and source identifiers. Labels are 0 for truthful and 1 for deceptive. MU3D sample identifiers such as `MU3D_BF001_NL` map to the provider-relative video path in `source_id`. DOLOS `sample_id` retains the source clip stem, while `participant_id` supplies the frozen within-person grouping.

The semantic representation is the 768-dimensional CLS embedding from [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base), followed by L2 normalization. The model revision is `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` and custom-code revision is `40ced75c3017eb27626c9d4ea981bde21a2662f4`. The input reader validates dtype, shape, finite values and unit norms. Embedding extraction is performed by the user. This repository supplies the downstream learning and evaluation pipeline.

```bash
python -m rel01 validate --dataset DOLOS --embeddings /path/to/dolos.npy
python -m rel01 validate --dataset MU3D --embeddings /path/to/mu3d.npy
```

The optional lexical experiment accepts an NPZ file containing `psy` with shape `[N,36]`, boolean `mask` with the same shape, and optional integer `reason` codes. Row order matches the semantic input. Column order is listed in `rel01/protocol/lexical_columns.json`. Missing measurements are represented by NaN with a false validity mask. The training-fold normalizer produces 36 standardized values and 36 validity flags. Running `--models all` requires these prepared measurements.

## Train from prepared features

The default `--models main` runs Query-only (`M0`), SHR-CoRel (`M2`) and independently trained OtherRef-CoRel (`C1-0`). The auxiliary setting `--models all` additionally runs lexical Query-only (`M1`), lexical SHR-CoRel (`M3`) and lexical OtherRef-CoRel (`C1-1`).

```bash
# A single outer-fold / seed group, including its four inner folds and outer refit.
python -m rel01 train --dataset DOLOS --embeddings /path/to/dolos.npy --outer 0 --seeds 20260722 --output /path/to/runs/dolos-main

# All five outer folds and all three seeds, sequentially.
python -m rel01 train --dataset DOLOS --embeddings /path/to/dolos.npy --output /path/to/runs/dolos-main
python -m rel01 train --dataset MU3D --embeddings /path/to/mu3d.npy --output /path/to/runs/mu3d-main

# The optional lexical experiments, using a separate output directory.
python -m rel01 train --dataset MU3D --embeddings /path/to/mu3d.npy --lexical /path/to/mu3d_lexical.npz --models all --output /path/to/runs/mu3d-all
```

Add `--device cuda` to select an available CUDA device. Choose an output directory outside the repository. Each group retains local checkpoints and prediction files and supports restarting with the same command and inputs. Main and auxiliary configurations use separate output directories. Input features, weights and run outputs are supplied or produced locally.

## Frozen protocol and parameters

`rel01/protocol/scopes.json` contains the exact 50 training, validation and test scopes for the two datasets. Each dataset has five participant-disjoint outer folds. Each outer training set is partitioned into four participant-disjoint inner folds. The sample mapping and scope lists fix the membership directly.

| Setting | Value |
|---|---|
| Seeds | 20260722, 20260723, 20260724 |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8 |
| Query / relation learning rate | 3e-4 / 1e-3 |
| Weight decay | 1e-3 on 2D weight matrices, zero on other parameters |
| Query batch size | 32 |
| Relation sampling | 8 participants and 4 queries per sampled reference |
| Query / relation maximum epochs | 100 / 80 |
| Query / relation early-stop patience | 15 / 12 |
| Selection metric | Participant/reference-balanced validation BCE |
| Improvement threshold | 1e-6 |
| Outer refit epochs | Ceiling of median of four inner selected epochs |
| Gradient norm clipping | 1.0 |
| Paired bootstrap | 5000 participant draws, seed 20260722 |

The relation stage considers epoch zero and freezes the selected query predictor. OtherRef-CoRel uses the same-capacity relation architecture with its own independently trained relation weights. OTHER donors are truthful responses from other participants in the current fitting set, matched by the absolute difference of `log(1+nlex)` with deterministic tie-breaking. The DOLOS same-source control also matches the source identifier and applies the frozen 0.25 length-gap cap.

`configs/frozen_parameters.json` records the scientific parameters. `rel01/protocol/implementation_contract.md` preserves the original V1.1 specification as provenance, including historical execution-stage statements. The public CLI binds execution to the feature arrays supplied by its user. Scientific model, loss, sampling, preprocessing and checkpoint calculations retain the frozen implementation.

## Evaluate the completed runs

```bash
python -m rel01 evaluate --run /path/to/runs/dolos-main --output /path/to/analysis/dolos.json
python -m rel01 evaluate --run /path/to/runs/mu3d-main --output /path/to/analysis/mu3d.json
```

Evaluation requires all five outer folds and three seeds for a dataset. It checks recorded prediction-file hashes and experiment identities. Metrics are computed within each participant/reference/seed condition, averaged over references and then seeds within each participant, and finally averaged equally across participants. Paired bootstrap intervals resample participants while retaining their reference and seed records.

The JSON report includes main metrics, paired comparisons, fixed-weight SELF/NULL/OTHER controls, the DOLOS strict same-source subset, and MU3D condition subsets. Ranking reversals use strictly positive versus strictly negative margins. Zero margins do not count as reversals. Seeds remain within-participant repetitions in the uncertainty calculation.

## Paper reference results

| Dataset | Query-only AP | SHR-CoRel AP | OtherRef-CoRel AP |
|---|---|---|---|
| DOLOS | 0.7323 | 0.7484 | 0.7272 |
| MU3D | 0.8892 | 0.9656 | 0.8944 |

`results/primary_metrics.csv` and `results/primary_paired_bootstrap.csv` contain the frozen full-precision summary values. Matching the numerical experiment requires the same prepared features, metadata and runtime conditions. Floating-point calculations can vary across CPU, CUDA and library versions.

## Data and licensing

The repository contains code, configuration, split membership and lightweight response identifiers and metadata needed to reproduce the protocol. Dataset access remains governed by the original providers. Large research assets are retained in the authors' research archive. The implementation is released under the MIT License. See [LICENSE](LICENSE). Dataset access terms apply to the underlying datasets.
