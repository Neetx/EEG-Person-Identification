# EEG Person Identification

This is the project I prepared for the Biometric Systems exam, it's an implementation of a paper called "EEG-based user identification system using 1D-convolutional long short-term memory neural networks".

https://www.sciencedirect.com/science/article/pii/S095741741930096X

THIS IS NOT MY RESEARCH, I programmed everything starting from their repository: https://github.com/ys4315/EEG-user-identification
It has matlab and python (tensorflow 1.1) components, so I wrote all the code using python 3.9 and tensorflow 2.6, extracting methods and parameters from the paper and the repository.

For now only the Convolutional Neural Network is implemented, but the paper proposal is a CLDNN (CNN + LSTM layers) and in the future I'll try to implement it too.

The project is a notebook for Google Colab and a python script for local execution.


## Dataset 

The dataset used is the same used by the paper: Physionet EEG motor Movement/Imagery dataset (https://physionet.org/content/eegmmidb/1.0.0/)
You have to upload this dataset to your google drive in a folder: "/content/gdrive/MyDrive/eeg_person_identification/eeg-motor-movementimagery-dataset-1.0.0/files/" to execute the notebook on Google Colab, otherwise you have to download the dataset on your local machine, create an output folder and write your paths in the code (for now)


## Results

The results are the same reported (97% accuracy)
