# CYP Inhibition — Starter Code

[Atom-JEPA](https://github.com/khelverskovp/atom-jepa) · [Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) · [Dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)

Starter repository for the DTU Deep Learning project **Predicting CYP Inhibition with Molecular Foundation Models** (project description provided separately as a PDF).

It gives you a simple, working fine-tuning setup to build on:

- **Atom-JEPA**: code for loading the pretrained Atom-JEPA model and fine-tuning it with a simple readout head.
- **Three baselines** to compare against: an MLP on Morgan fingerprints, on RDKit descriptors, and on both.
- **A simple fine-tuning protocol**: fixed train/validation/test splits, one training loop for all models, and the challenge metrics.

Your job is to extend it: add at least one more foundation model, try better readouts and training, and compare.

## What is in the repo

| Path | What it does |
|---|---|
| `prepare_data.py` | Downloads the data, generates 3D conformers and makes the 70/15/15 train/val/test split |
| `finetune.py` | Trains one model, picks the best epoch on validation, then evaluates on test |
| `models/atom_jepa.py` | Atom-JEPA encoder + readout head |
| `models/baselines.py` | Morgan / RDKit features + readout head |
| `models/readouts.py` | Readout heads (linear, MLP). **Add your own here** |
| `conf/` | [Hydra](https://hydra.cc) configs: `config.yaml` (shared), `model/` (one per model), `readout/` (one per head) |
| `prepare_data_cpu.sh`, `finetune.sh` | DTU HPC job scripts. Logs go to `outputs/` |

## Setup

Use Python 3.12 (on DTU, load a Python 3.12 module if needed).

```bash
git clone https://github.com/khelverskovp/dtu-cyp-project.git
cd dtu-cyp-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Run on DTU HPC

**1. Prepare the data (once).** This writes `data/` and takes a few minutes:

```bash
bsub < prepare_data_cpu.sh
```

**2. Fine-tune.** Pick the model and task with `-env` (default: Atom-JEPA, regression):

```bash
bsub < finetune.sh
bsub -env "MODEL=morgan_mlp, TASK=classification" < finetune.sh
bsub -env "MODEL=atom_jepa, OVERRIDES=readout=mlp train.epochs=10" < finetune.sh
```

`MODEL` is `atom_jepa`, `morgan_mlp`, `rdkit_mlp` or `morgan_rdkit_mlp`. `TASK` is `regression` or `classification`. `OVERRIDES` takes any Hydra settings, separated by spaces. Check jobs with `bjobs`.

## Tasks and results

- **Regression**: pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Main metric: macro ST-RAE (lower is better).
- **Classification**: time-dependent inhibition (TDI) for CYP2D6 and CYP3A4. Main metric: macro MCC (higher is better).

Missing labels are ignored in the loss. Each run is saved in `runs/<task>/<model>/<timestamp>/` with the validation history, the best checkpoint, `test_metrics.json` and `test_predictions.csv`. Use the same splits for every model, choose settings on validation, and use the test split only for the final comparison.

## Models and readouts

| `model=` | Input | Default readout |
|---|---|---|
| `atom_jepa` | 3D conformer → pretrained Atom-JEPA encoder (fine-tuned) | `linear` |
| `morgan_mlp` | Morgan count fingerprint | `mlp` |
| `rdkit_mlp` | 200 normalized RDKit descriptors | `mlp` |
| `morgan_rdkit_mlp` | Both, concatenated | `mlp` |

Any readout works with any model, e.g. `model=atom_jepa readout=mlp`. To add a readout, write a class in `models/readouts.py`, register it in `READOUTS` and add `conf/readout/<name>.yaml`. To add a model, follow the pattern in `models/` and add `conf/model/<name>.yaml`.

The baselines follow the MLP baselines in the [Atom-JEPA repository](https://github.com/khelverskovp/atom-jepa).

## Notes

- Two molecules (`OCNT-0453746`, `OCNT-0453782`) are excluded because their SMILES have impossible stereochemistry, so no 3D conformer exists. Both only have a negative CYP3A4 TDI label.
- Atom-JEPA runs in fp32 by default, because V100 GPUs do not support bf16 natively. On L40s or A100s, `model.precision=bf16` is faster.
