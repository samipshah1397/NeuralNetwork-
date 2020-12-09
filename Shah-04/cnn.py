# Shah, Samip
# 1001709873
# 2020_11_09
# Assignment-04-01


# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras.callbacks import History

class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network

        """
        self.model = None
        self.last_layer_defined = None
        self.flag = False

    def add_input_layer(self, shape=(2,),name="" ):
        """
         This function adds an input layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         Input layer is considered layer number 0 and it does not have any weights. Its purpose is to determine
         the shape of the input tensor and distribute it to the next layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        self.input = keras.Input(shape = shape, name = name)
        self.last_layer_defined = self.input
        return(None)


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        self.dense = keras.layers.Dense(units = num_nodes, activation = activation, name = name, trainable = trainable)(self.last_layer_defined)
        self.last_layer_defined = self.dense
        return(None)
        
         
    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        self.conv2d = keras.layers.Conv2D(filters = num_of_filters, kernel_size = kernel_size, padding = padding, strides = strides, activation = activation, name = name, trainable = trainable)(self.last_layer_defined)
        self.last_layer_defined = self.conv2d
        return(self.last_layer_defined)
         
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        self.maxpool2d = keras.layers.MaxPool2D(pool_size = pool_size, padding = padding, strides = strides, name = name)(self.last_layer_defined)
        self.last_layer_defined = self.maxpool2d
        return(self.last_layer_defined)
         
    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        self.flatten_layer = keras.layers.Flatten(name = name)(self.last_layer_defined)
        self.last_layer_defined = self.flatten_layer
        return(self.last_layer_defined)
         
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        if isinstance(layer_numbers, int):
            try:
                self.model.layers[layer_numbers].trainable = trainable_flag
            except:
                pass
        else:
            if len(layer_numbers) != 0:
                for number in layer_numbers:
                    try:
                        self.model.layers[number].trainable = trainable_flag
                    except:
                        pass
            else:
                if isinstance(layer_names, str):
                    try:
                        self.model.get_layer(layer_names).trainable = trainable_flag
                    except:
                        pass
                else:
                    if len(layer_names) != 0:
                        for name in layer_names:
                            try:
                                self.model.get_layer(name).trainable = trainable_flag
                            except:
                                pass
                    else:
                        pass


    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """
        self.model = keras.Model(self.input, self.last_layer_defined)

        if layer_number != None:
            try:
                weights = self.model.layers[layer_number].get_weights()[0]
                return(weights)
            except:
                pass
        else:
            try:
                weights = self.model.get_layer(layer_name).get_weights()[0]
                return(weights)
            except:
                pass


    def get_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """

        self.model = keras.Model(self.input, self.last_layer_defined)
        if layer_number != None:
            print('here')
            try:
                biases = self.model.layers[layer_number].get_weights()[1]
                return(biases)
            except:
                pass
        else:
            try:
                biases = self.model.get_layer(layer_name).get_weights()[1]
                return(biases)
            except:
                pass


    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        self.model = keras.Model(self.input, self.last_layer_defined)
        if layer_name != None:
            try:
                keras.backend.set_value(self.model.get_layer(layer_name).weights[0], weights)
            except:
                pass
        else:
            try:
                keras.backend.set_value(self.model.layers[layer_number].weights[0], weights)
            except:
                pass

    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        if self.model:
            pass
        else:
            self.model = keras.Model(self.input, self.last_layer_defined)
        if layer_name != None:
            try:
                keras.backend.set_value(self.model.get_layer(layer_name).weights[1], biases)
            except:
                pass

        else:
            try:
                keras.backend.set_value(self.model.layers[layer_number].weights[1], biases)
            except:
                pass
        
    def remove_last_layer(self):
        """
        This function removes a layer from the model.
        :return: removed layer
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        self.removed_layer = self.model._layers.pop()
        self.last_layer_defined = self.model.layers[-1].output
        return(self.removed_layer)

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        if model_name != "":
            if model_name == "VGG16":
                try:
                    from keras.applications.vgg16 import VGG16
                    self.model = VGG16()
                    self.last_layer_defined = self.model.get_layer("predictions")
                    return(self.model)
                except:
                    pass
            else:
                try:
                    from keras.applications.vgg19 import VGG19
                    self.model = VGG19()
                    self.last_layer_defined = self.model.get_layer("fc2").output
                    return(self.model)
                except:
                    pass
            
        else:
            try:
                self.model = keras.model.load_model(model_file_name)
                return(self.model)
            except:
                pass
        
    def save_model(self,model_file_name=""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        self.model.save(model_file_name)
        return(self.model)


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        self.model.compile(loss = loss)

    def set_metric(self,metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        self.model.compile(metrics = metric)
        return(None)

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        self.model.compile(optimizer = optimizer)
        return(None)


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        self.model = keras.Model(self.input, self.last_layer_defined)
        predict = self.model.predict(X)
        return(predict)

    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        self.model = keras.Model(self.input, self.last_layer_defined)
        _, scores = self.model.evaluate(X, y)
        return(scores)
         
    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        hist = History() 
        self.model = keras.Model(self.input, self.last_layer_defined)
        self.model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = num_epochs)
        return(hist.history['loss'])
         
if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    # print(my_cnn.model.summary())
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

    weights=my_cnn.get_weights_without_biases(layer_name="input")
    biases=my_cnn.get_biases(layer_number=0)
    print("input weights: ",None if weights is None else weights.shape,type(weights))
    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))
