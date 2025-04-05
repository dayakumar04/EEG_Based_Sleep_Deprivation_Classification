# Imports
import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_feature_importance(model, X_test, y_test, n_repeats=10, random_state=42):
    features = [
        'MeanAM',                         # 1
        'MeanBM',                         # 2
        'SpectralEntropy',                # 3
        'FrequencyCentroid',              # 4
        'MeanPeakAmplitude',              # 5
        'PeakFrequency',                  # 6
        'Skewness',                       # 7
        'Kurtosis',                       # 8
        'HjorthMobility',                 # 9
        'HjorthComplexity'                # 10
    ]

    bands = ['delta', 'theta', 'alpha', 'beta']  # 4 bands
    channels = ['O1', 'O2', 'Oz', 'PO3', 'PO7', 'POz', 'PO4', 'PO8']  # 8 channels

    feature_names = []
    for f in features:
        for b in bands:
            for c in channels:
                feature_names.append(f'{f}_{b}_{c}')

    result = permutation_importance(
    model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs = -1)

    importances = pd.Series(result.importances_mean, index=feature_names)

    return result, importances, feature_names