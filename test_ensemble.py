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
num_models = 20           #[1, 3, 5, 10, 20]
hidden_size = 256        #[64, 128, 256]
bagging_size = 25000       #[500, 1000, 5000, 15000, 25000] 10000?
#       baggingensemblenet --> in_size, hid_size, out_size, num_models / estimators
model = BaggingEnsembleNet(784, hidden_size, 10, num_models)
model.fit(
    x_train= x_train,
    y_train= y_train,
    bagging_size= bagging_size,
    val_loader= val_loader,
    learning_rate= 0.01,
    max_epochs= 50,
    val_acc_threshold= 0.95,
    batch_size= 256,
)

# evaluate the model on the test set
test_acc, test_loss = evaluate_model(model, test_loader)
print(f"Testing Parameters\n Hidden Size: {hidden_size}, Bagging Size: {bagging_size}, Num Models: {num_models}")
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")