# Shah, Samip
# 1001709873
# 2020-10-11
# Assignment-02-01

import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 



class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """

        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        # self._initialize_weights()
        self.initialize_weights()

    def hardlimit(self, ans):
        ans[ans < 0] = 0
        ans[ans > 0] = 1
        return ans
    
    def learn(self, alpha, sub, xit):
        return alpha * (np.dot(sub, xit))
        

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
             np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if self.weights.shape != W.shape:
            return -1
        self.weights = W
        
    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """

        return self.weights


    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        if self.transfer_function == "Hard_limit":
            # C = np.dot(self.weights, X)
            # C[C>=0] = 1
            # C[C<0] = 0
            # return C
            # temp = np.ones((1,X.shape[1]))
            # X = np.r_[temp, X]
            activation = np.dot(self.weights, X)
            a = self.hardlimit(activation)
            return (a)

        else:
            C = np.dot(self.weights, X)
            return C

        # newX = np.vstack((np.array([1 for column in range(X.shape[1])]), X))

        # results = np.dot(self.weights, newX)

        # if self.transfer_function == "Hard_limit":
        #     actualResults = np.where(results < 0, 0, 1)


        # return actualResults



    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        # numpy.linalg.pinv(a, rcond=1e-15, hermitian=False)[x]
        self.ps_inv = np.linalg.pinv(X)

        self.weights = np.dot(y, self.ps_inv)
        
    def hebb(self, rate):
        self.weights = self.weights + rate

    def filtered(self, gamma, rate):
        self.weights = (1 - gamma) * self.weights + rate
        
    def delta(self, rate):
        self.weights = self.weights + rate
        


    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
 # l = []
        # flag = 0
        # div = X.shape[1]//batch_size

        # with_ones = self.add_ones(X)

        # one_hot = np.zeros((y.shape[0],self.number_of_classes))
        # one_hot[np.arange(y.shape[0]),y] = 1
        # one_hot = one_hot.transpose()

        # if(with_ones.shape[1] % batch_size != 0):
        #     flag = 1

        # axis = 1
        # l = [batch_size*(i+1) for i  in range(math.floor(X.shape[axis] / batch_size))]

        # s = np.split(with_ones, l, axis = 1)
        # t = np.split(one_hot,l, axis = 1)

        #         if learning=="Delta":
        #             error = np.subtract(t[j], main_op)
        #             error = alpha*error
        #             final = np.dot(error, s[j].transpose())
        #             self.weights = self.weights + final
        #         elif learning=="Filtered":
        #             self.weights = (1-gamma)*self.weights + alpha*fil
        #         elif learning=="Unsupervised_hebb":
        #             un = alpha*main_op
        #             self.weights = self.weights + np.dot(un,s[j].transpose())

        l = len(X[0])
        for i in range(num_epochs):
            m = 0
            for j in range(int((math.ceil(l / batch_size)))):
                xi = X[:, m:m + batch_size]
                yo = y[:, m:m + batch_size]
                xit = np.transpose(xi)
                a = self.predict(xi)
                sub = np.subtract(yo, a)
                m += batch_size

                if learning == "Filtered":
                    # self.weights = (1-gamma)*self.weights + alpha*(np.dot(yo, xit))
                    rate = self.learn(alpha, yo, xit)
                #
                    self.filtered(gamma, rate)
                    
                elif learning == "Delta" or learning == "delta":
                    #           error = np.subtract(t[j], main_op)
                    #           error = alpha*error
                    #           final = np.dot(error, s[j].transpose())
                    # self.weights = self.weights + alpha*(np.dot(sub, xit))
                    rate = self.learn(alpha, sub, xit)
                    self.delta(rate)
                #
                else:
                    rate = self.learn(alpha,sub,xit)
                    self.hebb(rate)
    #
                    

        
        

    def error_fun(self, y, pred):
        difference_array = np.subtract(y, pred)
        squared_array = np.square(difference_array)
        errror = squared_array.mean()
        return errror

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        # decoded = []
        pred = self.predict(X)

        mean_squared_error = self.error_fun(y, pred)
        # pred = self.predict(X)
        # mean_square_error = (np.square(np.subtract(y, pred))).mean(None)
        # print(mean_squared_error)
        return mean_squared_error
        # with_ones = self.add_ones(X)

        # y_pred = np.dot(self.weights, with_ones)

        # if self.transfer_function == "Hard_limit":
        #     main_op = np.where(y_pred<=0, 0, 1)
        # if(self.transfer_function == "Linear"):
        #     main_op = y_pred
        # summation = 0
        # n = len(y)
        # for i in range (0,n):
        #     difference = y[i] - pred[i]
        #     squared_difference = difference**2
        #     summation = summation + squared_difference
        #     mean_square_error = summation/n
        # return mean_square_error
        # mean_square_error = mean_squared_error(y, pred)
        # mean_square_error = np.mean((y - pred)**2)

        # return mean_squared_error

        # for i in range(main_op.shape[1]):
        #     decoded.append(np.argmax(main_op[:,i]))

        # mean_square_error=round(1 - accuracy_score(list(y), decoded),2)

        # return mean_square_error