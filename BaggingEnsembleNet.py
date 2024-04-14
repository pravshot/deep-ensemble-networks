import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Bagging ensemble of simple neural networks
class BaggingEnsembleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_models):
        super(BaggingEnsembleNet, self).__init__()
        self.models = nn.ModuleList([SimpleNet(input_size, hidden_size, output_size) for _ in range(num_models)])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)

# training method for the bagging ensemble of simple neural networks
# TODO: implement this method

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# training method for the simple neural network
# will train for a maximum of max_epochs epochs or 
# until the validation accuracy exceeds val_acc_threshold
def train_simple_net(
    model,
    train_loader,
    val_loader,
    criterion = nn.CrossEntropyLoss(),
    learning_rate = 0.01,
    max_epochs = 50,
    val_acc_threshold = 0.7,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        # training
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, avg loss: {avg_loss}")
        
        # evaluating
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        
        val_acc = correct / total
        print("Validation accuracy:", val_acc)
        
        # check if we should stop training
        if val_acc > val_acc_threshold:
            print(f"Stopping training early, validation accuracy exceeded threshold of {val_acc_threshold}")
            break
