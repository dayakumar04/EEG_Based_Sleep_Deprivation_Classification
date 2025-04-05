from data_loader.data_loader import load_data, resample_data
from training.train_model import train_model
from training.hyperparameter_optimization import optimize_hyperparameters
from training.feature_importance import calculate_feature_importance
from models.model import get_model_and_param_dist
from utils.plotting import plot_cm_roc, plot_feature_importance


model_name = "rf"  # Choose from "svc", "xgb", "rf", or "lgb"

# Load data
print("Loading the data...")
X_train, X_test, y_train, y_test, labels = load_data(test_size=0.2, random_state=42)
X_train_small, y_train_small = resample_data(X_train, y_train, model_name, n_samples=2000, random_state=42)
print("Loaded the data.")

# Hyperparameter optimization
print("Optimizing hyperparameters...")
model, param_dist = get_model_and_param_dist(model_name)
best_model, best_params = optimize_hyperparameters(
    model, 
    model_name, 
    param_dist, 
    X_train_small, 
    y_train_small, 
    best_params=None,
    n_iter=20, 
    cv=5, 
    random_state=42
    )

# Train model
print("Starting training...")
fitted_model, y_pred, acc_score, classif_report = train_model(best_model, X_train, y_train, X_test, y_test)
print("Training finished.")
print(f"Test accuracy: {acc_score:.2f}")

#TODO: Save the model

# Feature importance
print("Calculating feature importance...")
result, importances, feature_names = calculate_feature_importance(fitted_model, X_test, y_test, n_repeats=10, random_state=42)

# Plot confusion matrix and ROC curve
save_dir = f"outputs/{model_name}"
plot_cm_roc(X_test, y_test, y_pred, fitted_model, labels, save_dir=save_dir)
plot_feature_importance(importances, feature_names, result, save_dir=save_dir)
