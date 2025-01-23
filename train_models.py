from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np

from helper import *

# from Experiments.HelperFiles.Code.helper import *

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_neural_net(X_train, y_train, num_classes):
    """
    Train a simple neural network on the given data.
    :param X_train: Training features
    :param y_train: Training labels
    :param num_classes: Number of classes for the output layer
    :return: Trained neural network model
    """
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # CrossEntropy requires Long type for labels

    # Create a simple dataset and DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)  # Output layer with num_classes nodes

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    print("Training complete.")
    return model


    def neural_net(x):
        output = net(x)[0,1] if x.shape[0]==1 else net(x)[:,1]
        return output

    def model(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return neural_net(x).detach().numpy()
    return model


def train_logreg(X_train, y_train, lime=False):
    logreg = LogisticRegression(random_state=0).fit(X_train, y_train)
    if lime:
        return logreg.predict_proba
    def model(x):
        return logreg.predict_proba(x)[:,1]
    return model

def train_rf(X_train, y_train, lime=False):
    rf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    if lime:
        return rf.predict_proba

    def model(x):
        return rf.predict_proba(x)[:,1]
    return model

def train_model(X_train, y_train, model_type, num_classes=None):
    """
    Train the model based on the specified model type.
    :param X_train: Training features
    :param y_train: Training labels
    :param model_type: Type of model to train (e.g., "nn")
    :param num_classes: Number of classes (for neural network models)
    :return: Trained model
    """
    if model_type == "nn":
        return train_neural_net(X_train, y_train, num_classes)
    else:
        raise ValueError("Unsupported model type. Only 'nn' is implemented.")
