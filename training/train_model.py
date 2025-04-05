# Imports
from sklearn.metrics import accuracy_score,  classification_report


def train_model(model, X_train, y_train, X_test, y_test):
    # Train the model on the entire training set with the best parameters
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)

    return model, y_pred, acc_score, classif_report