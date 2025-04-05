# Imports
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Function to load data
def load_data(test_size=0.2, random_state=42):
    f_v = scipy.io.loadmat('preprocessed_data/features.mat')
    f_v_data = f_v['features']

    labels_data = scipy.io.loadmat('preprocessed_data/labels.mat')
    labels = labels_data['labels'].squeeze()

    f_v_data = np.array(f_v_data)
    labels = np.array(labels).ravel()

    data_re = f_v_data.transpose(3, 0, 1, 2)
    data_re = data_re.reshape(9905, -1)
    print(data_re.shape)

    X_train, X_test, y_train, y_test = train_test_split(data_re, labels, test_size=test_size, random_state=random_state, stratify=labels)

    return X_train, X_test, y_train, y_test, labels

# Function to resample a smaller subset of data
def resample_data(X_train, y_train, model_name, n_samples=2000, random_state=42):
    if model_name == "rf" or model_name == "xgb":
        X_train_small, y_train_small = resample(X_train, y_train, n_samples=n_samples, stratify=y_train, random_state=random_state)
    else:
        X_train_small, y_train_small = resample(X_train, y_train, n_samples=n_samples, random_state=random_state)
    return X_train_small, y_train_small
