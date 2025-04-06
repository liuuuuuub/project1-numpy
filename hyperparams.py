from model import ThreeLayerNN
from train import train
from utils import accuracy
import numpy as np
from utils import load_cifar10

def search(X_train, y_train, X_val, y_val):
    results = {}
    best_acc = 0
    best_model = None
    best_params = None

    # 遍历不同的超参数组合
    for lr in [1e-1, 1e-2, 1e-3]:
        for reg in [1e-4, 1e-3, 1e-2]:
            for hidden in [64, 128]:
                print(f"Training with lr={lr}, reg={reg}, hidden={hidden}")

                # 创建模型并训练
                model = ThreeLayerNN(3072, hidden, 10, reg=reg)
                _, val_acc_log = train(model, X_train, y_train, X_val, y_val, epochs=5, lr=lr)

                # 获取最终的验证集准确率
                final_acc = val_acc_log[-1]
                results[(lr, reg, hidden)] = final_acc

                # 更新最佳准确率和模型
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_model = model
                    best_params = (lr, reg, hidden)

    # 输出最佳结果
    print("\nBest validation accuracy:", best_acc)
    print("Best hyperparameters: Learning rate = {}, Regularization = {}, Hidden units = {}".format(*best_params))

    return best_model, results


if __name__ == "__main__":
    # 这里你可以加载数据集并调用search函数
    X_train, y_train, X_val, y_val = load_cifar10('data/cifar-10-batches-py')
    # 运行超参数搜索
    best_model, results = search(X_train, y_train, X_val, y_val)

    # 输出每种超参数组合的验证准确度
    print("\nHyperparameter search results:")
    for params, acc in results.items():
        print(
            f"Learning rate = {params[0]}, Regularization = {params[1]}, Hidden units = {params[2]} => Validation Accuracy = {acc:.4f}")
