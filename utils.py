from keras.datasets import mnist
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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

def get_dataloader(x, y, batch_size=256, shuffle=True):
    """
    Create a DataLoader from input data
    """
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader