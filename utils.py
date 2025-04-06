import pickle
import os
import numpy as np


def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.astype('float32') / 255.0
        Y = np.array(Y)
        return X, Y
def load_cifar10(root):
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(root, f'data_batch_{b}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model.params, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

