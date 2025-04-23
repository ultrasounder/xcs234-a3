import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input

    Args:
        input_size (int): the dimension of inputs to be given to the network
        output_size (int): the dimension of the output
        n_layers (int): the number of hidden layers of the network
        size (int): the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
        Build a feed-forward network (multi-layer perceptron, or mlp) that maps
        input_size-dimensional vectors to output_size-dimensional vectors.
        It should have 'n_layers' hidden layers, each of 'size' units and followed
        by a ReLU nonlinearity. The final layer should be linear (no ReLU).

        Recall a hidden layer is a layer that occurs between the input and output
        layers of the network.

        As part of your implementation please make use of the following Pytorch
        functionalities:
        nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

    Hint:
        It is possible to create a list of nn.Modules and unpack these into nn.Sequential.
        For example:
            modules = []
            modules.append(nn.Linear(10, 10))
            modules.append(nn.Linear(10, 10))
            model = nn.Sequential(*modules)
    """
    ### START CODE HERE ###
    # Create a list of modules
    modules = []
    # handle the input dimensions for the first layer
    in_features = input_size
    
    # add n layers hidden layers
    for _ in range(n_layers):
        # add a linear layer with ReLU activation
        modules.append(nn.Linear(in_features, size))
        modules.append(nn.ReLU())
        # update the input dimensions for the next layer
        in_features = size
    
    # add the output layer
    modules.append(nn.Linear(in_features, output_size)) 
    
    # combine the modules into a sequential model
    return nn.Sequential(*modules)
    
    ### END CODE HERE ###
