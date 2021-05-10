import torch
from sklearn import model_selection


def train_test_split(X, Y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y)
    X_train = torch.from_numpy(x_train).float()
    X_test = torch.from_numpy(x_test).float()
    Y_train = torch.from_numpy(y_train).float()
    Y_test = torch.from_numpy(y_test).float()
    return X_train, X_test, Y_train, Y_test
