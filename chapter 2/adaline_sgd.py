import numpy as np

class AdalineSGD:
    """ADAptive LInear NEuron classifier.
        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        num_iter : int
            Passes over the training dataset.
        random_state : int
            Random number generator seed for random weight initialization.
        shuffle : bool (default: True)
            Shuffles training data every epoch if True to prevent
            cycles
        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting.
        b_ : Scalar
            Bias unit after fitting.
        losses_ : list
            Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.1, num_iter=50, random_state=42, shuffle=True):
            self.eta = eta
            self.num_iter = num_iter
            self.random_state = random_state
            self.shuffle = shuffle

    def fit(self, X, y):

        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.num_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            self.losses_.append(np.mean(losses))

        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
         
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                       size=m)
        self.bias_ = np.float64(0)
        self.weights_initialized = True

    def _update_weights(self, xi, y):
        """Apply Adaline learning rule to update the weights"""
        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = y - output
        
        self.w_ += self.eta * 2 * xi * error
        self.bias_ += self.eta * 2 * error

        loss = error**2
        return loss

    def net_input(self, X):
        return np.dot(X, self.w_) + self.bias_

    def activation(self, X):
          return X

    def predict(self, X):
          return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    