# Shah, Samip
# 1001709873
# 2020_10_25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None
        self.abc = []


    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """

        self.num_nodes = num_nodes
        if not self.abc:
            weights = np.random.randn(self.input_dimension, self.num_nodes)
        else:
            weights = np.random.randn(self.abc[-1]["weights"].shape[1], self.num_nodes)
        B = np.random.randn(self.num_nodes)
        self.weights.append(None)
        layer = {"transfer_function":transfer_function, "weights":weights, "B":B}
        self.abc.append(layer)
        
        
    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.abc[layer_number]["weights"]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.abc[layer_number]["B"]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.abc[layer_number]["weights"]  = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.abc[layer_number]["B"] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        x = X
        for i in range(len(self.abc)):
            a = tf.matmul(x,self.abc[i]["weights"]) + self.abc[i]["B"]
            if self.abc[i]["transfer_function"] == "Linear" or self.abc[i]["transfer_function"] == "linear":
                a = a
            elif self.abc[i]["transfer_function"] == "Relu" or self.abc[i]["transfer_function"] == "relu":
                a = tf.nn.relu(a, name='ReLU')
            else:
                a = tf.nn.sigmoid(a, name='sigmoid')
            x = a
        return x          
    
    def train_on_batch(self, x, y, learning_rate):
        r = []
        s = []
        for i in range(len(self.abc)):
            r.append(self.abc[i]["weights"])
            s.append(self.abc[i]["B"])
        with tf.GradientTape() as tape:
            pred = self.predict(x)
            loss = self.calculate_loss(y, pred)
            loss_weight, loss_bias = tape.gradient(loss, [r,s])
            for a in range(len(r)):
                weighter = learning_rate*loss_weight[a]
                biaser = learning_rate*loss_bias[a]
                r[a].assign_sub(weighter)
                s[a].assign_sub(biaser)  
            

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """

        self.y_train = tf.Variable(y_train)
        self.X_train = tf.Variable(X_train)
        for epochs in range(num_epochs):
            for j in range(0, np.shape(X_train)[0], batch_size):
                X_batcher = X_train[j:j+batch_size]
                Y_batcher = y_train[j:j+batch_size]
                self.train_on_batch(X_batcher, Y_batcher, alpha)


    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
    
        S = self.predict(X)
        S = np.argmax(S, axis = 1)
        sample = 0
        for i in range(len(S)):
            if y[i] != S[i]:
                sample+=1
        return(sample/len(S))


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        self.X_test = tf.Variable(X)

        self.Y_Predict = self.predict(X)
        
        prediction=tf.transpose(self.predict(X))
        print(prediction.shape)
        confusion_matrix=np.zeros((prediction.shape[0],prediction.shape[0]))
        for c,r in enumerate(y):
            correct_index=np.argmax(prediction[:,c])
            confusion_matrix[r][correct_index]+=1
        return confusion_matrix




