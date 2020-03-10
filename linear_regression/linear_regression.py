"""
https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
"""

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.stats as stats

############################
## Define hyperparameters ##
############################

# Training hyperparameters
learningRate = 0.01
epochs = 100

#####################
## Create data set ##
#####################

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


###########
## Model ##
###########

class linearRegression(torch.nn.Module):
    def __init__(self, inputDim, outputDim):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputDim, outputDim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Initialise the model
model = linearRegression(inputDim=1, outputDim=1)

###############
## Inference ##
###############

# Mean Squared Error as loss function
criterion = torch.nn.MSELoss() 

# Stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(x_train)) # no minibatches
    labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers at each epoch
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

#################################################
## Compare to linear regression fit with scipy ##
#################################################

# Extract weights from the NN model
params_nn = list(model.parameters())
params_nn = [ x.data.numpy().round(2) for x in params_nn ]
print("\nNN parameters:")
print(params_nn)

params_lr = stats.linregress(x_train[:,0], y_train[:,0])[:2]
print("\nLR parameters:")
print(params_lr)

##########
## Plot ##
##########

# with torch.no_grad(): # we don't need gradients in the testing phase
#     predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

# plt.clf()
# plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
# plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
# plt.legend(loc='best')
# plt.show()

