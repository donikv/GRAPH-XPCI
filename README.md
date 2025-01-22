# GRAPH-XPCI
GRAFT-XPCI: Dataset of synchrotron X-ray images for detection of acute cellular rejection after heart transplantation

## Dataset

This repository contains the code and description of the GRAFT-XPCI dataset used for classification of accute cellular rejection in heart transplantation. 
The dataset is organized into folders, with each folder containing images in an uncompressed 8-bit `.tif` format. 
These folders are categorized based on the date of imaging, providing a chronological structure that simplifies data navigation and management. 

To facilitate the use of the dataset across different evaluation protocols, `.csv` files are provided for each protocol.
The protocols and evaluation splits are designed such that there is no overlap between evaluation and training samples, to ensure no data leakage and fair comparison.
The whole dataset and the `csv` files are available [here](https://puh.srce.hr/s/f8p5fnxTfcH4HXy).

## Code structure

The code used to load the dataset, load pre-trained models, run experiments, and evaluate the results on the GRAFT-XPCI dataset (among other things) is structured as follows:

```
.
├── common -> folder containing utilities used for the evaluation, dataset, model definition...
├── environment.yml -> used for setting up the conda environment
├── records -> folder for pretrained models and storage of trained models
├── csv -> folder for storing csv files used for the evaluation procedures
├── config -> folder for storing .yml configuration files
├── notebooks -> collection of juypter notebooks used for testing and plotting
│   └── evaluation_protocols.ipynb -> used for generating results from the paper
├── training
│   ├── regression.py -> training for CNN models
│   ├── regression_vit.py -> training for ViT models
│   ├── regression_vitmae.py -> training for MAE models
│   ├── run_mae.py
│   └── runner.py -> main entry point
└── visualization -> 3D visualization of the data
    └── visualize3d.py
```

- To run the code, a conda environment needs to be created from the current folder using:
   - ```conda env create --name development -f environment.yml```
- An experiment is defined in the `.yml` file and is run with `training/runner.py` script that handles loading the arguments and calling the appropriate python module. The training is currently limited to one GPU when using `cuda`.
- Running an experiment will produce a log file, a graph of training dynamics, a `.pt` file containing model weights and all the info used in training. These files will be placed in the directory defined in the `log` argument. 
- The selection of the training and testing `csv` files is done using the `train-csv` and `test-csv` attributes. 
- Models and the datasets have to be downloaded from [here](https://puh.srce.hr/s/f8p5fnxTfcH4HXy).
	- Models need to be placed in the `records` folder, .csv files (including the folder structure) into the `csv` folder
	- Dataset can be placed anywhere on the disk, but the `path` argument that points to the root folder of the dataset needs to be changed in the .yml files or added to the training script as shown below

## Examples

Training a model from scratch on 1024x1024 images:
```
$ python training/runner.py config/regression_archive_resnet18.yml --args path=<PATH_TO_DATASET_ROOT>
```

Training a ViT model using pre-trained MAE (that can be downloaded from [here](https://puh.srce.hr/s/f8p5fnxTfcH4HXy), at `models/test_vit_mae_patches_new2/test.pt`, and should be placed in `records/test_vit_mae_patches_new2/test.pt` folder locally.)
```
$ python training/runner.py config/ViT_MAE_fresh_patches.yml --args path=<PATH_TO_PATCH_DATASET_ROOT>
```

-------------
 GRAFT-XPCI: Dataset of synchrotron X-ray images for detection of acute cellular rejection after heart transplantation © 2025 by Donik Vrsnak is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
