# CloudSegmentation
This project was done as part of a deep learning course at university.

## data_processing
- This folder contains the python files used to analyze and clean the 95-Cloud dataset

## data_sample
- Contains a single sample of the dataset including the input image patches for each channel and the corresponding corresponding ground truth image

## evaluation_scripts
- Contains two Matlab scripts that were used to evaluate the performance of the model on the 95-Cloud test set
- These two Matlab scripts are avaliable in github repo of the the 38-Cloud dataset: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset/tree/master/evaluation

## Attention-based_CNN_for_Cloud_Segmentation.pdf
- The paper explaining the dataset, the proposed model architecture, etc.
- [View the PDF](Attention-based_CNN_for_Cloud_Segmentation.pdf)

## cloud_detection.ipynb
- The file which contains all the finalized code for loading the processed 95-Cloud dataset, defining the model architecture, training the model, saving the model state, etc
- The file was run on Kaggle so the codes are written so that it loads dataset directory from the custom Kaggle dataset and saves all of the outputs to the Kaggle output directory
