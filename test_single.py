from BaggingEnsembleNet import SimpleNet
import numpy as np
from utils import load_mnist, get_dataloader

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = load_mnist()

# set up data loaders
train_loader = get_dataloader(
    x_train, np.eye(10)[y_train], batch_size=256, shuffle=True
)
test_loader = get_dataloader(x_test, np.eye(10)[y_test], batch_size=256, shuffle=False)

# train a simple neural network
model = SimpleNet(784, 256, 10)
val_acc, epochs = model.fit(
    train_loader, test_loader, learning_rate=0.01, val_acc_threshold=0.99, max_epochs=50
)
