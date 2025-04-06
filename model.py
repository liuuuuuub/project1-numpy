import numpy as np

class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', reg=0.0, lr=1e-3):
        self.params = {}
        self.reg = reg
        self.activation = activation
        self.lr = lr
        self._init_weights(input_dim, hidden_dim, output_dim)

    def _init_weights(self, D, H, C):
        # He initialization (for ReLU activation)
        self.params['W1'] = np.random.randn(D, H) * np.sqrt(2.0 / D)
        self.params['b1'] = np.zeros((1, H))
        self.params['W2'] = np.random.randn(H, C) * np.sqrt(2.0 / H)
        self.params['b2'] = np.zeros((1, C))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def _elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def forward(self, X):
        # Forward pass
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X.dot(W1) + b1
        if self.activation == 'relu':
            h1 = self._relu(z1)
        elif self.activation == 'leaky_relu':
            h1 = self._leaky_relu(z1)
        elif self.activation == 'elu':
            h1 = self._elu(z1)

        scores = h1.dot(W2) + b2
        return scores, h1

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Forward pass
        scores, h1 = self.forward(X)

        if y is None:
            return scores

        # Compute the loss
        probs = self._softmax(scores)
        N = X.shape[0]
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss

        # Backward pass
        grads = {}
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        grads['W2'] = h1.T.dot(dscores) + self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0, keepdims=True)
        dh1 = dscores.dot(W2.T)

        if self.activation == 'relu':
            dz1 = dh1 * self._relu_grad(h1)
        elif self.activation == 'leaky_relu':
            dz1 = dh1 * (h1 > 0) + 0.01 * (h1 <= 0)  # Leaky ReLU gradient
        elif self.activation == 'elu':
            dz1 = dh1 * (h1 > 0) + (h1 <= 0) * (h1 + 1)  # ELU gradient

        grads['W1'] = X.T.dot(dz1) + self.reg * W1
        grads['b1'] = np.sum(dz1, axis=0, keepdims=True)

        return loss, grads

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(self._softmax(scores), axis=1)