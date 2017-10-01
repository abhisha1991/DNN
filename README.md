# Deep Neural Network Implementation

This project attempts to build a neural network from scratch. It closely follows the coursework laid out in <a href="https://www.deeplearning.ai/">Deeplearning.ai</a>

The aim of this project is to give a step by step understanding of the implementation of the model building. 
Basically there are a few high level steps to keep in mind when implementing a NN:

1. Initialize parameters W and b for each layer in the NN
2. Reshaping the data to be fed as a single 64 x 64 x 3 vector (you guessed it! We're doing image classifications with this NN!)
3. Perform the forward pass by finding **Z = (W.T * X + b)** at each layer and then applying the relevant activation function 'g' on Z, giving us **A = g(Z)** as the output for layer 'L' and as the input for layer 'L+1'
4. Choosing the correct activations per layer - in this case, since we are performing a classication task with our NN, we perform **ReLu for L-1 layers and softmax for layer L**
5. Finding the cost after the forward propagation step - in this case, we find the logistical loss
6. Perform back propagation to find the derivatives for each of the layers so we can update the parameters W and b in order to minimize total loss
7. Repeat the entire process from 3 to 6 in minibatches across multiple epochs. 
8. Compare multiple models by varying hyperparameters (layer size, # layers, dropout probability, learning rate etc. in order to achieve maximum test accuracy)

## Requirements
1. Numpy
2. h5py
3.matplot lib

## Usage
Simply run main.py

## Credits
A big thank you to Coursera/Deeplearning.ai for putting a set of 5 courses together for enhancing my knowledge on deep learning. I am a big fan of Andrew Ng's simplistic teaching style and I can honestly say, he's shaped a lot of my understanding of these topics! 
Note - Some of the util code in the above code is provided by them.
