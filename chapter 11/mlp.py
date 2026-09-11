import numpy as np

class NeuralNetMLP:

    def __init__(self, num_classes, num_hidden, num_features, random_state=123):
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden = num_hidden

        rng = np.random.RandomState(random_state)

        self.w_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.b_h = np.zeros(num_hidden)

        self.w_out= rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.b_out = np.zeros(num_classes)

    def activation(self, x):
        return 1/(1+np.exp(-x))

    def one_hot(self, y, num_classes):
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    def forward(self, X):
        z_h = np.dot(X, self.w_h.T) + self.b_h
        a_h = self.activation(z_h)

        z_out = np.dot(a_h, self.w_out.T) + self.b_out
        a_out = self.activation(z_out)

        return a_h, a_out

    def backward(self, X, y, a_h, a_out):
        y_one_hot = self.one_hot(y, self.num_classes)

        # dimension: [num_examples, num_classes]
        d_loss__d_a_out = a_out - y_one_hot
        # dimension: [num_examples, num_classes]
        d_a_out__d_w_out = a_out * (1 - a_out)
        # dimension: [num_examples, num_classes]
        delta_output = d_loss__d_a_out * d_a_out__d_w_out
        # dimension: [num_examples, num_hidden]
        d_z_out__d_w_out = a_h

        d_loss_d_w_out = np.dot(delta_output.T, d_z_out__d_w_out)
        d_loss_d_b_out = np.sum(delta_output, axis=0)

        # dimension: [num_classes, num_hidden]
        d_z_out__d_a_h = self.w_out
        # dimension: [num_examples, num_hidden]
        d_loss__d_a_h = np.dot(delta_output, d_z_out__d_a_h)

        d_a_h__d_z_h = a_h * (1 - a_h)
        # dimension: [num_examples, num_features]
        d_z_h__d_w_h = X

        d_loss__d_w_h = np.dot((d_loss__d_a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum(d_loss__d_a_h * d_a_h__d_z_h, axis=0)

        return d_loss__d_w_h, d_loss__d_b_h, d_loss_d_w_out, d_loss_d_b_out