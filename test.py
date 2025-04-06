from model import ThreeLayerNN
from utils import load_cifar10, load_model, accuracy
import numpy as np

def test(model_path, X_test, y_test):
    model = ThreeLayerNN(3072, 128, 10)
    model.params = load_model(model_path)
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy(y_pred, y_test))


if __name__ == "__main__":
    # 加载CIFAR-10测试数据
    X_train, y_train, X_test, y_test = load_cifar10('data/cifar-10-batches-py')

    # 设定加载模型路径
    model_path = 'checkpoint/best_model.pkl'

    # 运行测试
    test(model_path, X_test, y_test)