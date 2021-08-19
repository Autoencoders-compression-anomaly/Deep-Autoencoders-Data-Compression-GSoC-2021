# Deep-Autoencoders-Data-Compression-GSoC-2021
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Gitter](https://badges.gitter.im/HEPAutoencoders/community.svg)](https://gitter.im/HEPAutoencoders/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

ML data compression of ATLAS trigger jet events using various deep autoencoders, with PyTorch and fastai python libraries.

This repository is developed by George Dialektakis, as a Google Summer of Code (GSoC) student

[Setup](#setup)

[Data extraction](#data-extraction)

## Setup
First, clone the latest version of the project to any directory of your choice:
```
git clone https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021.git
```
Install dependencies:
```
pip3 install -r requirements.txt
```


## Data extraction
The data that were used for this project can be downloaded from [CERN Open Data Portal](http://opendata.cern.ch/record/6010). The file that was used is: *00992A80-DF70-E211-9872-0026189437FE.root* under the filename *CMS_Run2012B_JetHT_AOD_22Jan2013-v1_20000_file_index.txt*. The data can then be loaded with `data_loader()`, which produces a pandas dataframe from the ROOT file.
