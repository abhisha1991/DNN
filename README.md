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

Here are some helpful images from the coursework which present a high level understanding of the process:

![alt text](https://github.com/aus2101/DNN/blob/master/images/1.png)

The above image gives an overall architecture of the process we talked about in points 1-8 above.


![alt text](https://github.com/aus2101/DNN/blob/master/images/2.PNG)

The above image talks about the different dimensions of Weights, Z values, Activation values and biases in each of the layers. This is probably the most important part to understand in the model! (Time to put your Linear Algebra on!). Try and work this out with a shallow NN - with 1 hidden layer and see if you're able to convince yourself the dimensions of your matrix multiplications match the above. Dont skip this step!

![alt text](https://github.com/aus2101/DNN/blob/master/images/4.PNG)

The above is the loss function we are trying to bring down as much as possible. Andrew Ng in his videos gives a very cogent explanation on why this is better than MSE and he convinces you well on why we use logistic loss for classification problems.

![alt text](https://github.com/aus2101/DNN/blob/master/images/5.PNG)

The above is a high level architecture of the flow of the NN. As you can see, we apply relu for L-1 layers and then a softmax for the last layer. This is because relu will allow the NN to train faster for L-1 layers as its derivative is 1 for positive values of Z. Softmax is used in the last layer because its equation spits out probabilities for each of the classes that add upto 1, so it makes our lives easier when we are trying to select the class with the highest probability.

![alt text](https://github.com/aus2101/DNN/blob/master/images/6.PNG)

This outlines the backward pass we perform in dnn_backprop.py - basically, we have to perform a single backprop step to find the gradients for softmax and then we perform L-1 backprops to find the gradients of the relu activation we had applied earlier. We keep track of each of these gradients for our parameters (W,b). Finally, we add them up and divide by the sample size 'm' after which we update our parameters for the next epoch iteration.

Finally after doing all this, you take your model with its existing state of weights and you predict the classes by iterating over the test set. For each prediction, we can keep track of a metric on whether we were able to successfully predict the image or not. Finally, report your accuracy score.

## Requirements
1. Numpy
2. h5py
3. matplot lib

## Usage
Simply run main.py

## Credits
A big thank you to Coursera/Deeplearning.ai for putting a set of 5 courses together for enhancing my knowledge on deep learning. I am a big fan of Andrew Ng's simplistic teaching style and I can honestly say, he's shaped a lot of my understanding of these topics! 
Note - Some of the util code in the above code is provided by them. All images above are from their tutorials.
