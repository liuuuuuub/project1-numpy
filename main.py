from model import ThreeLayerNN
from train import train
from test import test
from utils import load_cifar10
from visualize import plot_training, visualize_weights
import os

if not os.path.exists('checkpoint'):
    os.mkdir('checkpoint')

X_train, y_train, X_test, y_test = load_cifar10('data/cifar-10-batches-py')
X_val, y_val = X_train[45000:], y_train[45000:]
X_train, y_train = X_train[:45000], y_train[:45000]

model = ThreeLayerNN(3072, 128, 10, activation='relu', reg=1e-2)
train_loss, val_acc = train(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-2)
plot_training(train_loss, val_acc)
visualize_weights(model.params['W1'])
test('checkpoint/best_model.pkl', X_test, y_test)
