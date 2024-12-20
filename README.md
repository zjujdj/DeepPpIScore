# DeepPpIScore

This is the official implementation for the paper titled 'Harnessing Deep Statistical Potential for Biophysical Scoring of Protein-peptide Interactions'.

## Table of Contents
- [DeepPpIScore](#deepppiscore)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Evaluation of Peptide Binding Mode Prediction Based on the Well-established Unbound Set](#evaluation-of-peptide-binding-mode-prediction-based-on-the-well-established-unbound-set)
  - [Evaluation of Peptide Binding Mode Prediction Based on the Latest Bound Set](#evaluation-of-peptide-binding-mode-prediction-based-on-the-latest-bound-set)
  - [Comparison with AF-M 2.3 On the Peptide Binding Mode Prediction](#comparison-with-af-m-23-on-the-peptide-binding-mode-prediction)
  - [Conda Environment Reproduce](#conda-environment-reproduce)
    - [Create environment using yaml file provided in `./env` directory](#create-environment-using-yaml-file-provided-in-env-directory)
    - [Create environment using conda-packed `.tar.gz` file](#create-environment-using-conda-packed-targz-file)
  - [Training Structures and Evaluation Datasets](#training-structures-and-evaluation-datasets)
    - [Training Structures](#training-structures)
    - [PepSet](#pepset)
    - [BoundPep](#boundpep)
    - [PepBinding](#pepbinding)
    - [pMHCSet](#pmhcset)
  - [Usage](#usage)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Downloading ESM2 CheckPoint](#step-2-downloading-esm2-checkpoint)
    - [Step 3: Inference Example with the Weights Trained in This Study](#step-3-inference-example-with-the-weights-trained-in-this-study)
    - [Step 4: Model Re-training with the Training Structures Used in This Study](#step-4-model-re-training-with-the-training-structures-used-in-this-study)



## Introduction

Protein-peptide interactions (PpIs) play a critical role in major cellular processes. Recently, a number of machine learning (ML)-based methods have been developed to predict PpIs, but most of them rely heavily on sequence data, limiting their ability to capture the generalized molecular interactions in three-dimensional (3D) space, which is crucial for understanding protein-peptide binding mechanisms and advancing peptide therapeutics. Protein-peptide docking approaches provide a feasible way to generate the structures of PpIs, but they often suffer from low-precision scoring functions (SFs). To address this, we developed DeepPpIScore, a novel SF for PpIs that employs unsupervised geometric deep learning coupled with physics-inspired statistical potential. Trained solely on curated experimental structures without binding affinity data or classification labels, DeepPpIScore exhibits broad generalization across multiple tasks. Our comprehensive evaluations in bound and unbound peptide binding mode prediction, binding affinity prediction, and binding pair identification reveal that DeepPpIScore outperforms or matches state-of-the-art baselines, including popular protein-protein SFs, ML-based methods, and AlphaFold-Multimer 2.3 (AF-M 2.3). Notably, DeepPpIScore achieves superior results in peptide binding mode prediction compared to AF-M 2.3. More importantly, DeepPpIScore offers interpretability in terms of hotspot preferences at protein interfaces, physics-informed noncovalent interactions, and protein-peptide binding energies.

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig1.jpg)

## Evaluation of Peptide Binding Mode Prediction Based on the Well-established Unbound Set

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig2.jpg)

## Evaluation of Peptide Binding Mode Prediction Based on the Latest Bound Set

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig4.jpg)

## Comparison with AF-M 2.3 On the Peptide Binding Mode Prediction

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig5.jpg)

## Conda Environment Reproduce

### Create environment using yaml file provided in `./env` directory
Mamba Installation
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Then, the following code can be used to reproduce the conda environment:
```bash
mkdir -p ~/conda_env/DeepPpIScore
mamba env create --prefix=~/conda_env/DeepPpIScore --file ./env/DeepPpIScore.yaml
mamba activate ~/conda_env/DeepPpIScore
```

### Create environment using conda-packed `.tar.gz` file
Downloading conda-packed `.tar.gz` file from google dirve [DeepPpIScore.tar.gz](https://drive.google.com/file/d/1YEX5lwE3zd0gag3s_awReNNDPuNKPowt/view?usp=sharing), and then using the following code to reproduce the conda environment:

```bash
mkdir -p ~/conda_env/DeepPpIScore
tar -xzvf DeepPpIScore.tar.gz -C ~/conda_env/DeepPpIScore
mamba activate ~/conda_env/DeepPpIScore
conda-unpack
```

## Training Structures and Evaluation Datasets
### Training Structures
The training structures are available at [zenodo](https://zenodo.org/uploads/13881778).
### PepSet
Pepset is available at [PepSet Benchmark](http://cadd.zju.edu.cn/pepset/).
### BoundPep
The BoundPep structures are available at [zenodo](https://zenodo.org/uploads/13881778).
### PepBinding
PepBinding is available at [PepBinding](https://github.com/zjujdj/DeepPpIScore/blob/master/data/pdbbind2020_Ppi_binding_data_445.csv).
### pMHCSet
The pMHCSet structures are available at [zenodo](https://zenodo.org/uploads/13881778).


## Usage

The code was tested sucessfully on the basci environment equipped with `Nvidia Tesla V100 GPU Card`, `Python=3.9.13`, `CUDA=11.2`, `conda=24.3.0` and `mamba=1.5.8`

### Step 1: Clone the Repository

```bash
git clone https://github.com/zjujdj/DeepPpIScore.git
```

### Step 2: Downloading ESM2 CheckPoint

Downloading [esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt),   [esm2_t33_650M_UR50D-contact-regression.pt](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt) and put them in the `./data` directory

```bash
mv esm2_t33_650M_UR50D.pt esm2_t33_650M_UR50D-contact-regression.pt ./data
```

### Step 3: Inference Example with the Weights Trained in This Study

```bash
cd scripts
# the evaluation configurations can be set in file of model_inference_example.py
# submitting the following code to cpu node to generate graphs using multiprocessing first
python3 -u model_inference_example.py > model_inference_example_graph_gen.log
```
The generated graph files for this example were stored in the directory of `./data/temp_graphs_noH`

```bash
# then submitting the following code to gpu node to make predictions
python3 -u model_inference_example.py > model_inference_example.log
```
The prediction result was listed in directory of `./model_inference/DeepPpIScore/8.0/DeepPpIScore_8.0.csv` . For this csv file, four kinds of score were provided, namely 'cb-cb score', 'cb-cb norm score', 'min-min score' and 'min-min norm score', respectively where the norm score = score / sqrt(contacts). All the analysis in the paper was based on 'min-min score'.

### Step 4: Model Re-training with the Training Structures Used in This Study

Downloading the prepared training structures from google dirve [pepbdb_graphs_noH_pocket_topk30.zip](https://drive.google.com/file/d/1QNDU1Dj06FBCDUhtLPgRWEJzumukr7Ko/view?usp=drive_link) and [pepbdb_graphs_noH_ligand.zip](https://drive.google.com/file/d/1Y1zLU4ONfHp80zCYdVXOrhK3_4M0yP-m/view?usp=drive_link), and unzip them in the `./data` directory.

```bash
# unzip training structures
cd ./data
unzip pepbdb_graphs_noH_pocket_topk30.zip 
unzip pepbdb_graphs_noH_ligand.zip
# model training, the training configurations can be set in file of train_model.py
cd scripts
python3 -u train_model.py > train_model.log
```