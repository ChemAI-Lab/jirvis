# j-IR-vis: Vision model for Infrared Spectroscopy (IR) embeddings
[![DOI:](https://img.shields.io/badge/DOI-10.1063%2F5.0250837-blue)]()

This repository contains the code for reproducing the results for j-IR-vis, or to run inference on custom inputs.


## Dataset

### Sample Dataset
A [sample dataset](./data/sample) is included with this repository for preliminary use of jirvis. This dataset contains 32 random IR Spectroscopy images along with their functional group labels from both the simulated and experimental datasets.

### Custom Dataset
See [this tutorial](./README.md) for more information on how to incorporate your own custom dataset. 


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ChemAI-Lab/jirvis.git
cd jirvis
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f env.yaml
conda activate jirvis
```

## Using j-IR-vis
Adjust config files or use command line overrides as below.

### j-IR-vis Training
Using the experimental IR dataset. 
```bash
python3 scripts/train.py data=exp_ir
```
To use the simulated dataset, use `data=exp_ir` instead.

### Running Inference

```bash
python3 scripts/inference.py
```

# Reference
```latex
@article{molpipx,
    author = {Sondhi, Rudra and Chacko, Edwin R. and Vargas-Hernández, Rodrigo A.},
    title = {j-IR-vis: Vision model for Infrared spectroscopy embeddings},
    journal = {},
    volume = {},
    number = {},
    pages = {},
    year = {2025},
    month = {},
    issn = {},
    doi = {},
    url = {},
    eprint = {},
}
```