"""
https://nextjournal.com/gkoehler/pytorch-mnist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

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

#####################################
## Load data set using TorchVision ##
#####################################

# Training data
# train_loder is ???
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/ricard/test/pytorch/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

# Test data
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/Users/ricard/test/pytorch/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)

# - example_data is: torch.Size([1000, 1, 28, 28]), 
# 		This means we have 1000 examples of 28x28 pixels in grayscale
# - example_targets is: torch.Size([1000, 1, 28, 28])
# .
batch_idx, (example_data, example_targets) = next(examples)


#######################
## Build the network ##
#######################

# two 2-D convolutional layers followed by two fully-connected (or linear) layers.
# as activation function we'll choose rectified linear units
# aas a means of regularization we'll use two dropout layers

class Net(nn.Module):

    # Here we define the layers, not the connections yet.
    def __init__(self):
        super(Net, self).__init__()
        # 2D convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer
        self.conv2_drop = nn.Dropout2d()
        # Fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # The forward() pass defines the way we compute our output using the given layers and functions
    def forward(self, x):
    	# Convolution + Pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # ???? Reshape???
        x = x.view(-1, 320)


        # Fully connected layer
        x = self.fc1(x)

        # RELU Activation function
        x = F.relu(x)

        # Dropout layer
        # - self.training????
        x = F.dropout(x, training=self.training)

        # Fully connected layer
        x = self.fc2(x)

        # Output using a softmax
        out = F.log_softmax(x)
        return out


##############################################
## Initialize the network and the optimizer ##
##############################################

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

########################
## Training the model ##
########################

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):

	# network in train mode
  network.train()

  # loop over data
  for batch_idx, (data, target) in enumerate(train_loader):

		# Manually set the gradients to zero since PyTorch by default accumulates gradients. 
    optimizer.zero_grad()

    # do the NN operations
    output = network(data)

    # Compute negative log likelihood loss
    loss = F.nll_loss(output, target)

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

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

#############
## Iterate ##
#############

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

###################
## Plot training ##
###################

plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

###################
## Plot examples ##
###################

with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()
