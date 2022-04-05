# beatboxing-classification

Beatboxing classification using the Amateur Vocal Percussion (AVP) dataset. This is the code used for our final report of the MSCI 446 project.

# Environment Setup
Install conda and create env from given file:
```
conda env create -f environment.yml
```

# Dataset Setup

Download the dataset [here](https://zenodo.org/record/5036529#.Yi5Vln9KhH4) and unzip it.

## Pre-processing
We use the onset labels to split up the audio clips into individual utterances. To run this pre-processing, do:
```
python preprocess.py --root <DATASET_ROOT_DIR> --val_ratio <VALIDATION_SET_RATIO>
```

# Running CNN training
## Training script
To run the training, setup the config.yaml with the correct dataset path and desired hyperparameters then run:
```
python train.py config.yaml
```

## Sweeps
To produce our final model, we used Weights & Biases to run a hyperparameter sweep over the learning rate, weight decay, and eps parameters. To run the sweep, you first need to login using "wandb login" and entering your wandb API key. To start a sweep, first run:
```
wandb sweep sweep.yaml
```
A sweep ID will be printed out in the terminal. To then start training models, run:
```
wandb agent <SWEEP_ID>
```

# MFCC Learning
The rest of our experiments were done in the MFCC_learning.ipynb notebook. This includes all non-CNN supervised learning and all unsupervised leanring.
