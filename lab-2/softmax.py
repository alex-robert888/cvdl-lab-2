import numpy as np
from numpy.core.fromnumeric import size
from lab2.activations import softmax
from functools import reduce

from lab2.train import X_train

class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.__weights = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find np.random.randn userful here *0.001
        weights_height = self.__num_classes
        weights_width  = 1 + self.__input_shape[0] * self.__input_shape[1]
        self.__weights = 0.001 * np.random.randn(weights_height, weights_width)
        
        # Initialize the bias column with zeros
        self.__weights[:, -1] = np.zeros(weights_height)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        # remember about the bias t rick!
        # 1. apply the softmax function on the scores
        # 2. returned the normalized scores
        scores = self.__compute_output()
        return scores

    def predict(self, X: np.ndarray) -> int:
        label = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        # 1. compute the prediction by taking the argmax of the class scores
        output = self.__compute_output(with_softmax=False)
        label  = output.argmax()
        return label

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        steps = kwargs['steps'] if 'steps' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print(bs, reg_strength, steps, lr)

        # run mini-batch gradient descent
        for iteration in range(0, steps):
            print(f"mini-batch no. {iteration}")

            # TODO your code here
            # sample a batch of images from the training set
            # you might find np.random.choice useful
            indices = np.random.choice(len(X_train), bs)
            X_batch, y_batch = X_train[indices], y_train[indices]
            # compute the loss and dW
            dW = self.__compute_gradient(X_batch, y_batch)
            loss = self.__compute_loss(X_batch, y_batch, bs, reg_strength)
            # end TODO your code here
            # perform a parameter update
            self.__weights -= lr * dW
            # append the training loss, accuracy on the training set and accuracy on the test set to the history dict
            history.append(loss)

        return history


    def get_weights(self, img_shape):
        W = None
        # TODO your code here
        # 0. ignore the bias term
        # 1. reshape the weights to (*image_shape, num_classes)
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        with open(path, "rb") as f:
            self.__num_classes = np.load(f, allow_pickle=True)
            self.__input_shape = np.load(f, allow_pickle=True)
            self.__weights     = np.load(f, allow_pickle=True)
        
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find np.save useful for this
        # TODO your code here
        with open(path, "wb") as f:
            np.save(f, self.__num_classes)
            np.save(f, self.__input_shape)
            np.save(f, self.__weights)

        return True

    def __compute_output(self, X: np.ndarray, with_softmax: bool=True):
        # Compute the output column
        output = np.dot(X, self.__weights)
        output = softmax(output) if with_softmax else output 
        return output

    def __compute_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray):
        # Compute the gradient of the mini-batch
        X_batch_transposed                        = X_batch.transpose()
        one_hot                                   = np.zeros(y_batch.size, (self.__num_classes))
        one_hot[np.arange(y_batch.size), y_batch] = 1
        CT                                        = softmax(X_batch * self.__weights) - one_hot
        return np.dot(X_batch_transposed, CT)

    def __compute_loss(self, X_batch: np.ndarray, y_batch: np.ndarray, regularization_strength: float):
        average_loss = self.__compute_average_loss(X_batch, y_batch)
        regularization_term = self.__compute_regularization_term(regularization_strength)
        return average_loss + regularization_term 
        
    def __compute_average_loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        losses = np.vectorize(self.__compute_image_loss)(np.column_stack(X_batch, y_batch))
        return 1 / X_batch.size() * losses.sum()

    def __compute_image_loss(self, X_and_y):
        X, y = X_and_y
        output = self.__compute_output(X)
        return -1 * output[y] + np.log(np.sum(np.exp(output)))

    def __compute_regularization_term(self, regularization_strength):
        sum_of_weights_squares = sum(self.__weights ** 2)
        return regularization_strength * sum_of_weights_squares

