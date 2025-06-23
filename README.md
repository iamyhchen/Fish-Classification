# Training a CNN model for fish image classification
This repository contains the necessary scripts to train a CNN model for fish image classification using a Kaggle dataset ["*A Large Scale Fish Dataset*"](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset).

# Result on Test subset of CNN train
| model     |  	Accuracy  |  
|----------|----------|
| custom model     | 98.24%    | 
| vgg16 prtrained model     | 96.72%   |

# User guide
## Setup a virtual environment
```
python3 -m venv env_fish-classification
source env_fish-classification/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the dataset
Download the dataset on [Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) and put it in the `dataset` directory.
The project will be like this:
```
Fish-Classification
├── dataset
│   └── Fish_Dataset
│       ├── Black Sea Sprat
│       │   └── Black Sea Sprat
│       │       ├── 00001.png
│       │       └── ...
│       ├── Gilt-Head Bream
│       │   └── Gilt-Head Bream
│       │       ├── 00001.png
│       │       └── ...
│       ├── Hourse Mackerel
│       │   └── Hourse Mackerel
│       │       ├── 00001.png
│       │       └── ...
│       ├── Red Mullet
│       │   └── Red Mullet
│       │       ├── 00001.png
│       │       └── ...
│       ├── Red Sea Bream
│       │   └── Red Sea Bream
│       │       ├── 00001.png
│       │       └── ...
│       ├── Sea Bass
│       │   └── Sea Bass
│       │       ├── 00001.png
│       │       └── ...
│       ├── Shrimp
│       │   └── Shrimp
│       │       ├── 00001.png
│       │       └── ...
│       ├── Striped Red Mullet
│       │   └── Striped Red Mullet
│       │       ├── 00001.png
│       │       └── ...
│       └── Trout
│           └── Trout
│               ├── 00001.png
│               └── ...
├── custom.py
├── README.md
├── requirements.txt
├── split_dataset.py
└── vgg16.py
```
## Data preparetion
Run `split_dataset.py` to split the dataset into a training set, validation set, and test set.

## Start training
Run `custom.py` to train the CNN using a custom model.
```
python3 custom.py
```
Run `vgg16.py` to train the CNN using a vgg16 pretrained model.
```
python3 vgg16.py
```
The training results will be saved in the `exp` directory.