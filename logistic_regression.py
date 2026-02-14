# X has dimension (n_samples, n_features)
# W has dimension (n_features, 1)
# y has dimension (n_samples, 1)
# z = XW
# h = 1 / (1 + exp(-z))
# L = П(h ^ y * (1 - h) ^ (1 - y))
# l = Sum(y * ln(h) + (1 - y) * ln(1 - h)) - log likelihood
# J = -(1/m)l - cross-entropy loss function
# dJ/dW_j = dJ/dl * dl/dh * dh/dz * dz/dW
# dJ/dl_j = -(1/m)
# dl_j/dh_j = y_j/h_j - (1 - y_j)/(1 - h_j)
# dh_j/dz_j = h_j * (1 - h_j)
# dz_j/dW_j = x_j
# dJ/dW_j = -(1/m) * (y_j/h_j - (1 - y_j)/(1 - h_j)) * h_j * (1 - h_j) * x_j = -(1/m) * (y_j/h_j * h_j * (1 - h_j) * x_j - (1 - y_j)/(1 - h_j) * h_j * (1 - h_j) * x_j) =
# = -(1/m) * (y_j * (1 - h_j) * x_j - (1 - y_j) * h_j * x_j) = -(1/m) * (y_j * x_j - y_j * h_j * x_j - h_j * x_j + y_j * h_j * x_j) =
# = -(1/m) * (y_j - h_j) * x_j = (1/m) * (h_j - y_j) * x_j
# dJ/dW = 1/m * X_t * (h(XW) - y)

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_gradient(X, y, W):
    m = y.size

    return (1 / m) * (X.T @ (sigmoid(X @ W) - y))


def gradient_descent(X, y, lr=0.01, iter=1000, tolerance=1e-6):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # prepend bias term
    W = np.zeros(X_b.shape[1])
    for _ in range(iter):
        gradient = calculate_gradient(X_b, y, W)
        if np.linalg.norm(gradient) < tolerance:
            break

        W -= lr * gradient
    return W


def predict(X, W, threshold=0.5):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # prepend bias term
    return (sigmoid(X_b @ W) > threshold).astype(int)


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

W = gradient_descent(X_train_scaled, y_train)

y_pred_train = predict(X_train_scaled, W)
y_pred_test = predict(X_test_scaled, W)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# The sklearn LogisticRegression model
clf = LogisticRegression().fit(X_train_scaled, y_train)
print(clf.get_params())
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
