# DeepPpIScore: Harnessing Deep Statistical Potential for Biophysical Scoring of Protein-peptide Interactions

## Introduction

Protein-peptide interactions (PpIs) play a critical role in major cellular processes. Recently, a number of machine learning (ML)-based methods have been developed to predict PpIs, but most of them rely heavily on sequence data, limiting their ability to capture the generalized molecular interactions in three-dimensional (3D) space, which is crucial for understanding protein-peptide binding mechanisms and advancing peptide therapeutics. Protein-peptide docking approaches provide a feasible way to generate the structures of PpIs, but they often suffer from low-precision scoring functions (SFs). To address this, we developed DeepPpIScore, a novel SF for PpIs that employs unsupervised geometric deep learning coupled with physics-inspired statistical potential. Trained solely on curated experimental structures without binding affinity data or classification labels, DeepPpIScore exhibits broad generalization across multiple tasks. Our comprehensive evaluations in bound and unbound peptide binding mode prediction, binding affinity prediction, and binding pair identification reveal that DeepPpIScore outperforms or matches state-of-the-art baselines, including popular protein-protein SFs, ML-based methods, and AlphaFold-Multimer 2.3 (AF-M 2.3). Notably, DeepPpIScore achieves superior results in peptide binding mode prediction compared to AF-M 2.3. More importantly, DeepPpIScore offers interpretability in terms of hotspot preferences at protein interfaces, physics-informed noncovalent interactions, and protein-peptide binding energies.
![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig1.jpg)

## Evaluation of Peptide Binding Mode Prediction Based on the Well-established Unbound Set

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig2.jpg)

## Comparison with AF-M 2.3 On the Peptide Binding Mode Prediction

![Image text](https://github.com/zjujdj/DeepPpIScore/blob/master/figs/fig5.jpg)

## Conda Environment Reproduce

### **create environment using yaml file provided in `./env` directory**

The following commands can be used to reproduce the conda environment:

```bash
conda env create --prefix=/opt/conda_env/DeepPpIScore -f ./env/DeepPpIScore.yaml
```

## Usage

- **step 1: Clone the Repository**

```bash
git clone https://github.com/zjujdj/DeepPpIScore.git
```

- **step 2: Construction of Conda Environment**

```bash
conda env create --prefix=/opt/conda_env/DeepPpIScore -f ./env/DeepPpIScore.yaml
conda activate /opt/conda_env/DeepPpIScore
```

- **step 3: Downloading ESM2 CheckPoint**

Downloading [esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt),   [esm2_t33_650M_UR50D-contact-regression.pt](https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt) and put them in the `./data` directory

```bash
mv esm2_t33_650M_UR50D.pt esm2_t33_650M_UR50D-contact-regression.pt ./data
```

- **step 4: Inference Example**

```bash
cd scripts
# submitting the following code to cpu node to generate graphs
python3 -u model_inference_example.py > model_inference_example_graph_gen.log
# submitting the following code to gpu node to make predictions
python3 -u model_inference_example.py > model_inference_example.log
```

- **step 5: Model Re-training Using the Dasesets in this Study**

Downloading the prepared training structures [pepbdb_graphs_noH_pocket_topk30.zip](https://drive.google.com/file/d/1QNDU1Dj06FBCDUhtLPgRWEJzumukr7Ko/view?usp=drive_link) and [pepbdb_graphs_noH_ligand.zip](https://drive.google.com/file/d/1Y1zLU4ONfHp80zCYdVXOrhK3_4M0yP-m/view?usp=drive_link), unzip them in the `./data` directory; 

```bash
cd ./data
unzip pepbdb_graphs_noH_pocket_topk30.zip 
unzip pepbdb_graphs_noH_ligand.zip

cd scripts
python3 -u train_model.py > train_model.log
```
