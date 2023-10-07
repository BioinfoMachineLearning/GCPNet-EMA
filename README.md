<div align="center">

# GCPNet-EMA

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

Source code for the paper "Protein Structure Accuracy Estimation using Geometry-Complete Perceptron Networks".

## Contents

- [Installation](#installation)
- [GCPNet for protein structure EMA (GCPNet-EMA)](#gcpnet-for-protein-structure-ema-gcpnet-ema)
  - [How to prepare data for GCPNet-EMA](#how-to-prepare-data-for-gcpnet-ema)
  - [How to train GCPNet-EMA](#how-to-train-gcpnet-ema)
  - [How to evaluate GCPNet-EMA](#how-to-evaluate-gcpnet-ema)
  - [How to predict lDDT scores for new protein structures using GCPNet-EMA](#how-to-predict-lddt-scores-for-protein-structures-using-gcpnet-ema)
- [For developers](#for-developers)
- [Acknowledgements](#acknowledgements)
- [Citations](#citations)

## Installation

Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies

```bash
# clone project
git clone https://github.com/BioinfoMachineLearning/GCPNet-EMA
cd GCPNet-EMA

# create conda environment
mamba env create -f environment.yaml
conda activate GCPNet-EMA  # NOTE: one still needs to use `conda` to (de)activate environments

# install local project as package
pip3 install -e .
```

## GCPNet for protein structure EMA (`GCPNet-EMA`)

### How to prepare data for `GCPNet-EMA`

Download training and evaluation data

```bash
cd data/EMA/
wget https://zenodo.org/record/8150859/files/ema_decoy_model.tar.gz
wget https://zenodo.org/record/8150859/files/ema_true_model.tar.gz
tar -xzf ema_decoy_model.tar.gz
tar -xzf ema_true_model.tar.gz
cd ../../  # head back to the root project directory
```

### How to train `GCPNet-EMA`

Train a model for the estimation of protein structure model accuracy (**EMA**) task

```bash
python3 src/train.py experiment=gcpnet_ema.yaml
```

### How to evaluate `GCPNet-EMA`

Reproduce our results for the EMA task

```bash
ema_model_1_ckpt_path="checkpoints/EMA/model_1.ckpt"
ema_model_2_ckpt_path="checkpoints/EMA/model_2.ckpt"
ema_model_3_ckpt_path="checkpoints/EMA/model_3.ckpt"

python3 src/eval.py datamodule=ema model=gcpnet_ema logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$ema_model_1_ckpt_path"
python3 src/eval.py datamodule=ema model=gcpnet_ema logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$ema_model_2_ckpt_path"
python3 src/eval.py datamodule=ema model=gcpnet_ema logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$ema_model_3_ckpt_path"
```

```bash
EMA Model 1
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Test metric           ┃          DataLoader 0          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test/PerModelMAE        │      0.04894806072115898       │
│        test/PerModelMSE        │      0.004262289963662624      │
│  test/PerModelPearsonCorrCoef  │       0.8362738490104675       │
│       test/PerResidueMAE       │      0.06654192507266998       │
│       test/PerResidueMSE       │      0.009298641234636307      │
│ test/PerResiduePearsonCorrCoef │       0.7442569732666016       │
│           test/loss            │      0.005294517148286104      │
└────────────────────────────────┴────────────────────────────────┘

EMA Model 2
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Test metric           ┃          DataLoader 0          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test/PerModelMAE        │      0.04955434799194336       │
│        test/PerModelMSE        │      0.004251933190971613      │
│  test/PerModelPearsonCorrCoef  │       0.841285228729248        │
│       test/PerResidueMAE       │      0.06787651032209396       │
│       test/PerResidueMSE       │      0.009320290759205818      │
│ test/PerResiduePearsonCorrCoef │       0.7426220774650574       │
│           test/loss            │      0.005294565111398697      │
└────────────────────────────────┴────────────────────────────────┘

EMA Model 3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Test metric           ┃          DataLoader 0          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test/PerModelMAE        │      0.05056113377213478       │
│        test/PerModelMSE        │      0.004722739569842815      │
│  test/PerModelPearsonCorrCoef  │       0.8154276013374329       │
│       test/PerResidueMAE       │      0.07143402099609375       │
│       test/PerResidueMSE       │      0.01017170213162899       │
│ test/PerResiduePearsonCorrCoef │       0.7132763266563416       │
│           test/loss            │      0.005769775714725256      │
└────────────────────────────────┴────────────────────────────────┘
```

### How to predict lDDT scores for protein structures using `GCPNet-EMA`

Predict per-residue and per-model lDDT scores for computationally-predicted (e.g., AlphaFold 2) protein structure decoys

```bash
ema_model_ckpt_path="checkpoints/EMA/model_1.ckpt"
predict_batch_size=1  # adjust as desired according to available GPU memory
num_workers=0  # note: required when initially processing new PDB file inputs, due to ESM's GPU usage

python3 src/predict.py model=gcpnet_ema datamodule=ema datamodule.predict_input_dir=$MY_INPUT_PDB_DIR datamodule.predict_true_dir=$MY_OPTIONAL_TRUE_PDB_DIR datamodule.predict_output_dir=$MY_OUTPUTS_DIR datamodule.predict_batch_size=$predict_batch_size datamodule.num_workers=$num_workers logger=csv trainer.accelerator=gpu trainer.devices=1 ckpt_path="$ema_model_ckpt_path"
```

For example, one can predict per-residue and per-model lDDT scores for a batch of tertiary protein structure inputs, `6W6VE.pdb` and `6W77K.pdb` within `data/EMA/examples/decoy_model`, as follows

```bash
python3 src/predict.py model=gcpnet_ema datamodule=ema datamodule.predict_input_dir=data/EMA/examples/decoy_model datamodule.predict_output_dir=data/EMA/examples/outputs datamodule.predict_batch_size=1 datamodule.num_workers=0 datamodule.python_exec_path="$HOME"/mambaforge/envs/gcpnet/bin/python datamodule.lddt_exec_path="$HOME"/mambaforge/envs/gcpnet/bin/lddt datamodule.pdbtools_dir="$HOME"/mambaforge/envs/gcpnet/lib/python3.9/site-packages/pdbtools/ logger=csv trainer.accelerator=gpu trainer.devices=[0] ckpt_path=checkpoints/EMA/model_1.ckpt
```

**Note**: After running the above command, an output CSV containing metadata for the predictions will be located at `logs/predict/runs/YYYY-MM-DD_HH-MM-SS/predict_YYYYMMDD_HHMMSS_rank_0_predictions.csv`, with text substitutions for the time at which the above command was completed. This CSV will contain a column called `predicted_annotated_pdb_filepath` that identifies the temporary location of each input PDB file after annotating it with GCPNet-EMA's predicted lDDT scores for each residue. If a directory containing ground-truth PDB files corresponding one-to-one with the inputs in `datamodule.predict_input_dir` is provided as `datamodule.predict_true_dir`, then metrics and PDB annotation filepaths will also be reported in the output CSV to quantitatively and qualitatively describe how well GCPNet-EMA was able to improve upon AlphaFold's initial per-residue plDDT values.

## For developers

Set up `pre-commit` (one time only) for automatic code linting and formatting upon each `git commit`

```bash
pre-commit install
```

Manually reformat all files in the project, as desired

```bash
pre-commit run -a
```

Update dependencies in `environment.yml`

```bash
mamba env export > env.yaml # e.g., run this after installing new dependencies locally
diff environment.yaml env.yaml # note the differences and copy accepted changes back into `environment.yaml`
rm env.yaml # clean up temporary environment file
```

## Acknowledgements

GCPNet-EMA builds upon the source code and data from the following project(s):
* [EnQA](https://github.com/BioinfoMachineLearning/EnQA)
* [GCPNet](https://github.com/BioinfoMachineLearning/GCPNet)
* [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

We thank all their contributors and maintainers!

## Citing this work

If you use the code or data associated with this project, or otherwise find this work useful, please cite:

```bibtex
@article{morehead2023gcpnet_ema,
  title={Protein Structure Accuracy Estimation using Geometry-Complete Perceptron Networks},
  author={Morehead, Alex and Cheng, Jianlin},
  year={2023}
}
```