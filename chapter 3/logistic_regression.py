import numpy as np

class LogisticRegressionGD:
    """Gradient Descent for logistic Regression
        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        num_iter : int
            Passes over the training dataset.
        random_state : int
            Random number generator seed for random weight initialization.
        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting.
        b_ : Scalar
            Bias unit after fitting.
        losses_ : list
            Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.1, num_iter=50, random_state=42):
            self.eta = eta
            self.num_iter = num_iter
            self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])
        self.bias_ = np.float64(0)
        self.losses_ = []

        for _ in range(self.num_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_ += self.eta * 2 * X.T.dot(errors) / errors.shape[0]
            self.bias_ += self.eta * 2 * errors.mean()
            loss = -(y*np.log(output) + (1-y)*np.log(1-output)).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.bias_

    def activation(self, X):
          return 1 / (1 + np.exp(-X))

    def predict(self, X):
          return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
