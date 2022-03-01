# README

This is the code repository for paper "A Novel Semi-supervised Training Framework Guided by Two Sources of Uncertainties for Medical Image Segmentation"

The model training details can be found in `train_panc.py` file in `train` function. 

Folder `data_lists` specifies the train test split for pancreas dataset and left atrium dataset.

Folder `preprocess` contains code used to preprocess pancreas dataset.

Folder `trained_models` contains the model whose results are presented in our paper. 

Files with suffix "panc" means it uses pancreas dataset while suffix "LA" means left atrium dataset

If you'd like to train the model from scratch, you can run either `train_panc.py` or `train_LA.py`. You may need uncomment the line invoking pretrain function to get a pretrained model first and prepare corresponding datasets and modify the dataset path in the code.

