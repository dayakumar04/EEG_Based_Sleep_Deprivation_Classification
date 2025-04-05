import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os


# Function to plot confusion matrix and ROC curve
def plot_cm_roc(X_test, y_test, y_pred, model, labels, save_dir):
  conf_matrix = confusion_matrix(y_test, y_pred)

  plt.figure(figsize=(6, 5))
  ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                  annot_kws={"size": 14})

  plt.xlabel("Predicted Label", fontsize=14 )
  plt.ylabel("True Label", fontsize=14)
  plt.xticks(fontsize=14 )  
  plt.yticks(fontsize=14)  

  cbar = ax.collections[0].colorbar
  cbar.set_label("Count", fontsize=14) 
  cbar.ax.tick_params(labelsize=12)
  
  os.makedirs(save_dir, exist_ok=True)
  if save_dir:
     plt.savefig(f"{save_dir}/confusion_matrix.png")
  else:
    plt.show()

  if len(np.unique(labels)) == 2:
    y_test_binarized = label_binarize(y_test, classes=np.unique(labels))
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_binarized, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right",fontsize=14)
    plt.xticks(fontsize=14 ) 
    plt.yticks(fontsize=14)
    plt.grid()
    if save_dir:
      plt.savefig(f"{save_dir}/roc_curve.png")
    else:
      plt.show()
  else:
    print("ROC curve can only be plotted for binary classification.")


# Function to plot feature importance
def plot_feature_importance(importances, feature_names, result, save_dir):
    top_10_features = importances.abs().nlargest(10)

    top_10_indices = np.array([feature_names.index(f) for f in top_10_features.index])

    df = pd.DataFrame({
        "Feature": top_10_features.index,
        "Importance": top_10_features.values,
        "Error": result.importances_std[top_10_indices]
    })

    df = df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9,5))

    sns.barplot(data=df, x="Importance", y="Feature", errorbar=None, ax=ax, palette="viridis")

    for i, (imp, err) in enumerate(zip(df["Importance"], df["Error"])):
        ax.errorbar(imp, i, xerr=err, color='black', capsize=4)

    ax.set_xlabel("Mean accuracy decrease", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)

    plt.xticks(fontsize=13 )
    plt.yticks(fontsize=13)

    fig.tight_layout()
    plt.grid()

    os.makedirs(save_dir, exist_ok=True)
    if save_dir:
      plt.savefig(f"{save_dir}/feature_importances.png")
    else:
      plt.show()