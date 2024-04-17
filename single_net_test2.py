# test network on mnist dataset
from BaggingEnsembleNet import SimpleNet, BaggingEnsembleNet
from keras.datasets import mnist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

def load_mnist():
  '''
  Loads, reshapes, and normalizes the data
  '''
  (x_train, y_train), (x_test, y_test) = mnist.load_data() # loads MNIST data
  x_train = np.reshape(x_train, (len(x_train), 28*28))  # reformat to 768-d vectors
  x_test = np.reshape(x_test, (len(x_test), 28*28))
  maxval = x_train.max()
  x_train = x_train/maxval  # normalize values to range from 0 to 1
  x_test = x_test/maxval
  return (x_train, y_train), (x_test, y_test)

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = load_mnist()
x_train = x_train[:5000]
y_train = y_train[:5000]

# set up data loaders
trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(np.eye(10)[y_train]))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(np.eye(10)[y_test]))
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# train a simple neural network
model = SimpleNet(784, 256, 10)
val_acc, epochs = model.fit(
    train_loader,
    test_loader,
    learning_rate=0.01,
    val_acc_threshold=0.99,
    max_epochs=50
)

print(val_acc, epochs)

# model2 = BaggingEnsembleNet(784, 256, 10, 10)
# print(model2.forward(torch.Tensor(x_train[:2])))