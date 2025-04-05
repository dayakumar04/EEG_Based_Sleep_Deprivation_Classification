# Imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Function to optimize hyperparameters
def optimize_hyperparameters(estimator, estimator_name, param_distributions, X_train_small, y_train_small, best_params=None, scoring="accuracy", n_iter=20, cv=5, random_state=42):
    if best_params is None:
        print("Initializing RandomizedSearchCV...")
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            scoring=scoring,
            n_iter=n_iter,
            cv=cv,
            verbose=0,
            n_jobs=-1,
            random_state=random_state
        )

        print("Starting RandomizedSearchCV...")
        random_search.fit(X_train_small, y_train_small)
        print("RandomizedSearchCV complete!")

        best_params = random_search.best_params_

    if estimator_name.lower() == "svc":
        best_model = SVC(**best_params, random_state=42, probability=True)
    elif estimator_name.lower() == "xgb":
        best_model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42)
    elif estimator_name.lower() == "rf":
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif estimator_name.lower() == "lgb":
        best_model = lgb.LGBMClassifier(**best_params, random_state=42)
    else:
        raise ValueError("Invalid estimator name. Please choose from 'svc', 'xgb', 'rf', or 'lgb'.")

    print(f"Best Parameters: {best_params}")
    if not best_params:
        print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")
    return best_model, best_params