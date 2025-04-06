from model import ThreeLayerNN
from utils import load_cifar10, accuracy, save_model
import numpy as np
import os

def train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=200, lr=1e-3, lr_decay=0.95, momentum=0.9):
    N = X_train.shape[0]
    best_acc = 0
    train_loss_log = []
    val_acc_log = []
    velocity = {param: np.zeros_like(model.params[param]) for param in model.params}

    for epoch in range(epochs):
        idx = np.arange(N)
        np.random.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        losses = []

        for i in range(0, N, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss, grads = model.loss(X_batch, y_batch)
            losses.append(loss)

            # 更新参数时添加动量
            for param in model.params:
                velocity[param] = momentum * velocity[param] - lr * grads[param]
                model.params[param] += velocity[param]

        val_acc = accuracy(model.predict(X_val), y_val)
        avg_loss = np.mean(losses)
        train_loss_log.append(avg_loss)
        val_acc_log.append(val_acc)
        print(f"Epoch {epoch + 1}, lr={lr:.5f}, loss={avg_loss:.4f}, val acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, 'checkpoint/best_model.pkl')
        lr *= lr_decay
    return train_loss_log, val_acc_log