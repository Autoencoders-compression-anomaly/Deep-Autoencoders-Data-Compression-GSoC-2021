# Deep-Autoencoders-Data-Compression-GSoC-2021
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Gitter](https://badges.gitter.im/HEPAutoencoders/community.svg)](https://gitter.im/HEPAutoencoders/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

ML data compression of ATLAS trigger jet events using various deep autoencoders, with PyTorch and fastai python libraries.

This repository is developed by George Dialektakis, as a Google Summer of Code (GSoC) student

[Setup](#setup)

[Running the code](#running-the-code)

[Data extraction](#data-extraction)

[Project Structure Description](#project-structure-description)

## Setup
First, clone the latest version of the project to any directory of your choice:
```
git clone https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021.git
```
Install dependencies:
```
pip3 install -r requirements.txt
```

## Running the code
```
usage: python main.py [--use_vae] [--use_sae] [--l1] [--epochs] [--custom_norm]
                      [--num_variables] [--plot]

optional arguments:
  --use_vae            whether to use Variational AE (default: False)
  --use_sae            whether to use Sparse AE (default: False)
  --l1                 whether to use L1 loss or KL-divergence in the Sparse AE (default: True)
  --epochs             number of epochs to train (default: 50)
  --custom_norm        whether to normalize all variables with min_max scaler or also use custom normalization for 4-momentum (default: False)
  --num_variables      number of variables we want to compress (either 19 or 24) (default: 24)
  --plot               whether to make plots (default: False)
```
Example:

```
python main.py --use_sae True --epochs 30 --num_variables 19 --plot True
```
The above command will train the Sparse Autoencoder for 30 epochs to compress the 19D data and will make plots of the input and preprocessed data.

## Data extraction
The data that were used for this project can be downloaded from [CERN Open Data Portal](http://opendata.cern.ch/record/6010). The file that was used is: *00992A80-DF70-E211-9872-0026189437FE.root* under the filename *CMS_Run2012B_JetHT_AOD_22Jan2013-v1_20000_file_index.txt*. The data can then be loaded with `data_loader()`, which produces a pandas dataframe from the ROOT file.

## Project Structure Description
- `data_loader.py` loads the data from a ROOT file and creates a pandas dataframe.

- `data_processing.py` makes all the necessary preprocessing steps for our data (filtering, normalization, train-test split)

- `create_plots.py` holds all the necessary functions to plot the initial, preprocessed and reconstructed data

- `autoencoders/` holds the implementation of three different Autoencoder types we considered (Standard AE, Sparse AE, Variational AE)

- `evaluate.py` performs the evaluation of the autoencoder on the test data in terms of MSE, RMSE loss, and residuals

- `main.py` is the main script which runs the whole code

You can find more details about this project as well as the experimental analysis and the results in `report.pdf`
