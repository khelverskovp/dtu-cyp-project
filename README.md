# CYP Inhibition — Starter Code

**[Leaderboard](https://khelverskovp.github.io/dtu-cyp-leaderboard/)** · [Atom-JEPA](https://github.com/khelverskovp/atom-jepa) · [Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) · [Dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)

Starter repository for the DTU Deep Learning project **Predicting CYP Inhibition with Molecular Foundation Models** (project description provided separately as a PDF).

It gives you a simple, working fine-tuning setup to build on:

- **Atom-JEPA**: code for loading the pretrained Atom-JEPA model and fine-tuning it with a simple readout head.
- **Three baselines** to compare against: an MLP on Morgan fingerprints, on RDKit descriptors, and on both.
- **A simple fine-tuning protocol**: fixed train/validation/test splits, one training loop for all models, and the challenge metrics.

Your job is to extend it: add at least one more foundation model, try better readouts and training, and compare. You can submit your test predictions to the [project leaderboard](https://khelverskovp.github.io/dtu-cyp-leaderboard/) to see how you rank.

## What is in the repo

| Path | What it does |
|---|---|
| `prepare_data.py` | Downloads the two datasets, generates 3D conformers and splits each into 70/15/15 train/val/test |
| `finetune.py` | Trains one model, picks the best epoch on validation, then evaluates on test |
| `models/atom_jepa.py` | Atom-JEPA encoder + readout head |
| `models/baselines.py` | Morgan / RDKit features + readout head |
| `models/readouts.py` | Readout heads (linear, MLP). **Add your own here** |
| `conf/` | [Hydra](https://hydra.cc) configs: `config.yaml` (shared), `model/` (one per model), `readout/` (one per head) |
| `prepare_data.sh`, `finetune.sh` | DTU HPC job scripts. Logs go to `outputs/` |

## Setup

Use Python 3.12.

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

**1. Prepare the data (once).** This writes `data/regression/` and `data/classification/` (each with `train.csv`, `val.csv`, `test.csv`) and takes a few minutes:

```bash
bsub < prepare_data.sh
```

**2. Fine-tune.** Pick the model and task with `-env` (default: Atom-JEPA, regression):

```bash
bsub < finetune.sh
bsub -env "MODEL=morgan_mlp, TASK=classification" < finetune.sh
bsub -env "MODEL=atom_jepa, OVERRIDES=readout=linear train.epochs=10" < finetune.sh
```

`MODEL` is `atom_jepa`, `morgan_mlp`, `rdkit_mlp` or `morgan_rdkit_mlp`. `TASK` is `regression` or `classification`. `OVERRIDES` takes any Hydra settings, separated by spaces. Check jobs with `bjobs`.

## Tasks and results

There are two separate datasets from the [challenge data](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), one per task:

- **Regression** (`cyp-challenge-TRAIN_inhibition.csv`): pIC50 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Leaderboard score: MA-ST-RAE, the soft-threshold relative absolute error averaged over the four enzymes (lower is better).
- **Classification** (`cyp-challenge-TRAIN_TDI.csv`): time-dependent inhibition (TDI) for CYP2D6 and CYP3A4. Leaderboard score: MCC, averaged over the two enzymes (higher is better).

Many molecules appear in both datasets; they are always in the same split for both tasks, so you can safely combine the tasks. Missing labels are ignored in the loss. Each run is saved in `runs/<task>/<model>/<timestamp>/` with the validation history, the best checkpoint, `test_metrics.json` and `test_predictions.csv`. Use the same splits for every model, choose settings on validation, and use the test split only for the final comparison.

## Leaderboard

The original challenge ends before this project starts, so we run our own leaderboard on this repo's test split: **https://khelverskovp.github.io/dtu-cyp-leaderboard/**. There is one board for regression and one for classification, each showing the best submission per team.

**How to submit**

1. Choose your final model on the **validation** set.
2. Take the `test_predictions.csv` from that run (`runs/<task>/<model>/<timestamp>/`).
3. [Open a submission](https://github.com/khelverskovp/dtu-cyp-leaderboard/issues/new?template=submission.yml) (needs a GitHub account): enter your team name, the task and a short model description, and drag the CSV into the form.
4. A bot comments your score on the issue within a few minutes and updates the leaderboard.

**Rules**

- **Fine-tune and tune on the validation set, never on the test set.** Choose models, hyperparameters and epochs by validation scores only. Submit test predictions only for models you have already chosen.
- **One submission per team per task per day** (Copenhagen time). Rejected submissions do not count.
- **Use the splits from `prepare_data.py` with the default seed.** Don't train on test molecules or look at their labels. `data/<task>/test.csv` contains the labels and the data is public, so the leaderboard relies on your honesty.
- **Use the same team name** for all of your team's submissions.

## Models and readouts

| `model=` | Input | Default readout |
|---|---|---|
| `atom_jepa` | 3D conformer → pretrained Atom-JEPA encoder (fine-tuned) | `mlp` |
| `morgan_mlp` | Morgan count fingerprint | `mlp` |
| `rdkit_mlp` | 200 normalized RDKit descriptors | `mlp` |
| `morgan_rdkit_mlp` | Both, concatenated | `mlp` |

Any readout works with any model, e.g. `model=atom_jepa readout=linear`. To add a readout, write a class in `models/readouts.py`, register it in `READOUTS` and add `conf/readout/<name>.yaml`. To add a model, follow the pattern in `models/` and add `conf/model/<name>.yaml`.

The baselines follow the MLP baselines in the [Atom-JEPA repository](https://github.com/khelverskovp/atom-jepa).

## Notes

- Two molecules (`OCNT-0453746`, `OCNT-0453782`) are excluded because their SMILES have impossible stereochemistry, so no 3D conformer exists. Both only have a negative CYP3A4 TDI label.
- Atom-JEPA runs in fp32 by default, because V100 GPUs do not support bf16 natively. On L40s or A100s, `model.precision=bf16` is faster.
