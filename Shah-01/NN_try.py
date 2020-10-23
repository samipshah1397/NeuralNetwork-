# 1000-123-456
# 2020-09-27
# Assignment-01-01

import numpy as np
from math import exp


class SingleLayerNN(object):
    
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """

        self.input_dimensions = input_dimensions
        self.number_of_nodes=number_of_nodes
        self.initialize_weights()
   
    def initialize_weights(self,seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        # W = np.random.rand((input_dimensions,number_of_nodes))*np.sqrt(1/(ni+no))
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions+1)
        print(self.weights)
        print("No of classes:",number_of_nodes)
        
    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        self.weights = np.zeros((self.number_of_nodes,self.input_dimensions+1))
        print(self.weights)
        
    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights
        
    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
       # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: X}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.input_dimensions):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.weights[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a
        
        # row = []
        # for i in range (0,self.weights.shape[0]):
        #     activation = []
        #     for j in range (0,X.shape[1]):
        #         summation = np.dot(X[:,j],((self.weights[i,:self.weights.shape[1]-1].T))) + self.weights[i][self.weights.shape[1]-1]
        #         if summation > 0:
        #             activation.append(1)
        #         else:
        #             activation.append(0)            
        #     row.append(activation)
        # return(row)
        #raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")
        
        # temp = np.ones((1,X.shape[1]))
        # X = np.r_[temp, X]
        # activation = np.dot(self.weights,X)
        # activation[activation < 0] = 0
        # activation[activation > 0] = 1
        # return(activation)

        # return np.where((np.dot(X, self.weights[1:]) + self.weights[0]) >= 0.0, 1, 0)

        # value = 0
        # for x in range(len(X)):
        #     # Add the value of the inputs
        #     value += X[x] * self.weights[x]

        # # Add the value of bias
        # value += self.bias * self.weights[-1]

        # # Put value into the SIGMOID equation
        # return float(1/(1+exp(-value)))

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None

        """
        for i in range(num_epochs):
            err = 0
            for xi, target in zip(X, Y):
                delta_w = alpha * (target - self.predict(xi))
                self.weight[1:] += delta_w * xi
                self.weight[0] += delta_w
                err += int(delta_w != 0.0)
            self.errors.append(err)
        return self

    #     x_bias = np.ones((X.shape[0],1)) # bias term (nSamplesx1): pass the input value 1
    #     x = np.hstack((np.round(X / 255.0), x_bias)) # feature space: (nSamplesx785)
    #     y = 1*(Y==self.weights) + -1*(Y!=self.weights) # (nSamplesx1): +1 or -1
    
#         ones_array = np.ones(X.shape[1])
#         X  = np.insert(X, 0,ones_array, axis = 0)
#         for i in range(num_epochs):
#             for k in range(len(X[0])):
#                 O = np.dot(self.weights,X[:,k])
# #                 
#                 O[O>=0] = 1
#                 O[O<0] = 0
# #                 
#                 E = Y[ : ,k]-O
#                 E = np.asmatrix(E)
#                 E = np.transpose(E)
# #                 
#                 c = X[:, k]
# #                 
#                 c = np.asmatrix(c)
# #                 
#                 self.weights = self.weights + alpha*(np.dot(E, c))
    
    # # Learn and update weights (Gradient descent)
    # #nEpoch = 50, learnRate = 0.001
    #     for iter in range(num_epochs):
    #         y_hat = np.sign(np.dot(X, self.weights)) # (nSamplesx785)x(785x1)=(nSamplesx1)        
    #         updateOrNot = (y_hat != y)
    #         self.weights += alpha * np.dot(x.T, np.multiply(y, updateOrNot).astype(float))

    #     return self.weights # (nSamplesx1)

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
#         L = self.predict(X)
#         count = 0
#         total = len(X[0])
# #         print(total)
#         for k in range(len(X[0])):
#             xl = L[ : , k]
#             xl = np.asmatrix(xl)
#             yl = np.asmatrix(Y)
#             yl = yl[ : , k]
# #             
#             c = np.array_equal(np.asarray(xl).flatten(),np.asarray(yl).flatten())
# #             print(c)
#             if c:
#                 count+=1
#         a = (((total-count)/total)*(100))
# #         
#         return a

if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())