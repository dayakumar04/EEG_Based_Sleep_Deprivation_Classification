from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform

def get_model_and_param_dist(estimator_name, eval_metric="logloss", random_state=42):
    if estimator_name.lower() == "svc":
        model = SVC(probability=True, random_state=random_state)
        param_dist = {
            'C': uniform(0.1, 10),
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 0.01, 0.001]
        }
    elif estimator_name.lower() == "xgb":
        model = XGBClassifier(eval_metric=eval_metric, random_state=random_state)
        param_dist = {
            'n_estimators': [50, 100, 200],  # Number of trees
            'learning_rate': uniform(0.01, 0.2),  # Learning rate
            'max_depth': [3, 5, 7],  # Depth of each tree
            'subsample': uniform(0.7, 0.3),  # Sampling ratio for training data
            'colsample_bytree': uniform(0.7, 0.3)  # Sampling ratio for features
        }
    elif estimator_name.lower() == "rf":
        model = RandomForestClassifier(random_state=random_state)
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 50],
            'criterion' : ["gini", "entropy", "log_loss"],
            'max_features' : ["sqrt", "log2", None],
            'bootstrap': [True, False]
        }
    elif estimator_name.lower() == "lgb":
        model = lgb.LGBMClassifier()
        param_dist = {
            'num_leaves': [10, 20, 31, 40, 50],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_freq': [1, 3, 5, 7],
            'boosting_type': ['gbdt', 'dart']
        }
    else:
        raise ValueError("Invalid estimator name. Please choose from 'svc', 'xgb', 'rf', or 'lgb'.")
    
    return model, param_dist