"""
https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder/linear-autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Number of hidden units
nhidden = 30

# Number of epochs
n_epochs = 1

# Batch size
batch_size_train = 64
batch_size_test = 1000

# Training parameters
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Set random seed
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

###################
## Load data set ##
###################

# convert data to torch.FloatTensor
root = "/Users/ricard/test/pytorch/autoencoder_mnist/files"

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
  
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test)

#######################
## Build the network ##
#######################

class Autoencoder(nn.Module):

    # Here we define the layers, not the connections yet.
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # input -> hidden, with ReLU activation function
        x = F.relu(self.encoder(x))
        # hidden -> output, with sigmoid activation function (to match the input ranges, as data is normalised from 0 to 1)
        x = F.sigmoid(self.decoder(x))
        return x


##############################################
## Initialize the network and the optimizer ##
##############################################

network = Autoencoder(input_dim=28*28, hidden_dim=nhidden)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

########################
## Training the model ##
########################

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Specify loss
criterion = nn.MSELoss()

def train(epoch):

	# network in train mode
  network.train()

  # loop over data
  for batch_idx, (data, _) in enumerate(train_loader):

		# Manually set the gradients to zero since PyTorch by default accumulates gradients. 
    optimizer.zero_grad()

    # Flatten image from (...,28,28) to (...,28,28)
    data = data.view(data.size(0), -1)

    # do the NN operations
    output = network(data)

    # Compute data reconstruction MSE loss
    loss = criterion(output, data)

    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print statistics
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append( (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)) )

#############
## Iterate ##
#############

# test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  # test()

###################
## Plot training ##
###################

plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Loss')
plt.show()


###################
## Plot examples ##
###################

# with torch.no_grad():
#   output = network(example_data)

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Prediction: {}".format(
#     output.data.max(1, keepdim=True)[1][i].item()))
#   plt.xticks([])
#   plt.yticks([])

# plt.show()
