# beatboxing-classification

Beatboxing classification using the Amateur Vocal Percussion (AVP) dataset.

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

# Running training
To run the training, setup the config.yaml with the correct dataset path and desired hyperparameters then run:
```
python train.py config.yaml
```
