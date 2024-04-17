from BaggingEnsembleNet import BaggingEnsembleNet, evaluate_model
import numpy as np
from utils import load_mnist, get_dataloader

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = load_mnist()
x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]
# set up data loaders
val_loader = get_dataloader(x_val, np.eye(10)[y_val], batch_size=256, shuffle=False)
test_loader = get_dataloader(x_test, np.eye(10)[y_test], batch_size=256, shuffle=False)
# train a bagging ensemble of simple neural networks
model = BaggingEnsembleNet(784, 256, 10, 15)
model.fit(
    x_train= x_train,
    y_train= y_train,
    bagging_size= 7500,
    val_loader= val_loader,
    learning_rate= 0.01,
    max_epochs= 50,
    val_acc_threshold= 0.95,
    batch_size= 256,
)

# evaluate the model on the test set
test_acc, test_loss = evaluate_model(model, test_loader)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")