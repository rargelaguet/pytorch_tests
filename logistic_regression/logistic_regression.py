"""
https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
"""

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import scipy.stats as stats

############################
## Define hyperparameters ##
############################

# Training hyperparameters
learningRate = 0.01
epochs = 100

def sigmoid(X):
    return np.divide(1.,1.+np.exp(-X))

#####################
## Create data set ##
#####################

x_train = np.arange(-10,10,0.5).reshape(-1, 1).astype(np.float32)
y_train = sigmoid(x_train).round(0).reshape(-1, 1).astype(np.float32)


###########
## Model ##
###########

class LogisticRegression(torch.nn.Module):
    def __init__(self, inputDim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(inputDim,1)

    def forward(self, x):
        out = F.sigmoid(self.linear(x))
        return out

# Initialise the model
model = LogisticRegression(inputDim=1)

###############
## Inference ##
###############

# Binary cross entropy as loss function
criterion = torch.nn.BCELoss(size_average=True)


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

##################################################
## Compare to linear regression fit with scikit ##
##################################################

# Extract weights from the NN model
params_nn = list(model.parameters())
params_nn = [ x.data.numpy().round(2) for x in params_nn ]
print("\nNN parameters:")
print(params_nn)

# params_lr = stats.linregress(x_train[:,0], y_train[:,0])[:2]
# print("\nLR parameters:")
# print(params_lr)

##########
## Plot ##
##########

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted_continuous = model(Variable(torch.from_numpy(x_train))).data.numpy()
    predicted_binary = model(Variable(torch.from_numpy(x_train))).data.numpy().round(0)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted_continuous, '--', label='Continuous predictions', alpha=0.5)
plt.plot(x_train, predicted_binary, 'x', label='Binary predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()

