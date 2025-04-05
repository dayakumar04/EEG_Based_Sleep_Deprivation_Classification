This repository contains code for classifying Acute Sleep Deprivation using ML models trained on EEG-based spectral features.

#### Dataset 
This repository uses an open source dataset. The dataset provides resting-state EEG data (eyes open, partially eyes closed) from 71 participants who underwent the following two experiments - normal sleep (NS-session1) and  sleep deprivation (SD-session2). 
For this study, we used the eyes open data files. Data from participant 1 was not included due to missing data files. 
The dataset can be downloaded from the following link - https://openneuro.org/datasets/ds004902/versions/1.0.5
Citation: Chuqin Xiang and Xinrui Fan and Duo Bai and Ke Lv and Xu Lei (2024). A Resting-state EEG Dataset for Sleep Deprivation. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds004902.v1.0.5

#### Preprocessing 
The raw EEG data files are preprocessed using the EEGLAB toolbox in MATLAB. To preprocess the data, follow these steps:
1.	Copy all the folders containing the raw EEG data in the preprocessed_data folder.
2.	Run preprocessed_data/preprocessing_and_data_claening.m
3.	Run preprocessed_data/epoching.m
4.	Run preprocessed_data/feature_extraction.m
This saves the extracted features and labels in the preprocessed_data folder.

#### Training the models
The code supports four ML models – SVC, RF, XGBoost and LightGBM. To train the models, first run the preprocessing code in MATLAB which will save the features.mat and labels.mat files in the preprocessed_data folder.

The models are trained using the scikit learn library in Python. To train the models enter the name of the model you want to train in the main.py file, and then run the main.py file.

The optimal hyperparameters for each model are also provided in the optimal_parameters.txt file.
