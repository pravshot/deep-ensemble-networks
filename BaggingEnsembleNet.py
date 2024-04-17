import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


# Bagging ensemble of simple neural networks
class BaggingEnsembleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_models):
        super(BaggingEnsembleNet, self).__init__()
        self.models = nn.ModuleList(
            [SimpleNet(input_size, hidden_size, output_size) for _ in range(num_models)]
        )

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    # training method for the bagging ensemble of simple neural networks
    def fit(
        self,
        x_train,
        y_train,
        bagging_size,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.01,
        max_epochs=50,
        val_acc_threshold=0.6,
        batch_size=256,
    ):
        for i, model in enumerate(self.models):
            # sample a subset of the training data (with replacement)
            idxs = np.random.choice(len(x_train), bagging_size, replace=True)
            x_train_subset = x_train[idxs]
            y_train_subset = y_train[idxs]
            train_loader = DataLoader(
                TensorDataset(
                    Tensor(x_train_subset), Tensor(np.eye(10)[y_train_subset])
                ),
                batch_size=batch_size,
                shuffle=True,
            )
            val_acc, epoch = model.fit(
                train_loader,
                val_loader,
                criterion,
                learning_rate,
                max_epochs,
                val_acc_threshold,
                log=False,
            )
            print(
                f"Model {i+1} trained for {epoch} epochs, validation accuracy: {val_acc}"
            )


# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)

    # will train for a maximum of max_epochs epochs or
    # until the validation accuracy exceeds val_acc_threshold
    def fit(
        self,
        train_loader,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.01,
        max_epochs=50,
        val_acc_threshold=0.9,
        log=True,
    ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(max_epochs):
            # training
            self.train()
            total_loss = 0
            n = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                n += len(labels)

            avg_loss = total_loss / n
            if log:
                print(f"Epoch {epoch+1}, avg loss: {avg_loss}")

            # evaluating
            val_acc, val_loss = evaluate_model(self, val_loader)
            if log:
                print("Validation accuracy:", val_acc)

            # check if we should stop training
            if val_acc > val_acc_threshold:
                if log:
                    print(
                        f"Stopping training early, validation accuracy exceeded threshold of {val_acc_threshold}"
                    )
                return val_acc, epoch + 1
        return val_acc, epoch + 1


# evaluate a model on a dataloader and return acc, loss
def evaluate_model(model, loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    N = 0
    acc = 0
    loss = 0
    with torch.set_grad_enabled(False):
        for inputs, targets in loader:
            outputs = model(inputs)
            # Compute sum of correct labels
            y_pred = np.argmax(outputs.cpu().numpy(), axis=1)
            y_gt = np.argmax(targets.numpy(), axis=1)
            acc += np.sum(y_pred == y_gt)
            N += len(targets)
            # Compute loss
            loss += criterion(outputs, targets).item() * len(targets)
    loss /= N
    acc /= N
    return acc, loss
