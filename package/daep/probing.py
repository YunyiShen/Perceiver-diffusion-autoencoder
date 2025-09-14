import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def make_fewshot_loaders(test_dict, training_proportion=0.5, batch_size=64, names = ["encode", "types"]):
    X = test_dict[names[0]]
    y = test_dict[names[1]]

    X_support, y_support = [], []
    X_query, y_query = [], []

    # few-shot split per class
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        #breakpoint()
        idx_train, idx_test = train_test_split(idx, train_size=training_proportion, random_state=42)

        X_support.append(X[idx_train])
        y_support.append(y[idx_train])
        X_query.append(X[idx_test])
        y_query.append(y[idx_test])

    X_support = np.concatenate(X_support)
    y_support = np.concatenate(y_support)
    X_query = np.concatenate(X_query)
    y_query = np.concatenate(y_query)

    # wrap into loaders
    support_loader = DataLoader(TensorDataset(torch.tensor(X_support, dtype=torch.float32),
                                              torch.tensor(y_support, dtype=torch.long)),
                                batch_size=batch_size, shuffle=True)

    query_loader = DataLoader(TensorDataset(torch.tensor(X_query, dtype=torch.float32),
                                            torch.tensor(y_query, dtype=torch.long)),
                              batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    return support_loader, query_loader, input_dim, num_classes



def make_dataloaders(train_dict, test_dict, batch_size=128, names = ["encode", "types"]):
    """Wrap encodings + labels into PyTorch DataLoaders."""
    X_train = torch.tensor(train_dict[names[0]], dtype=torch.float32)
    y_train = torch.tensor(train_dict[names[1]], dtype=torch.long)

    X_test = torch.tensor(test_dict[names[0]], dtype=torch.float32)
    y_test = torch.tensor(test_dict[names[1]], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape[1], len(torch.unique(y_train))


class LinearProbe(nn.Module):
    """Linear classifier for probing."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    """Tiny MLP probe (1 hidden layer)."""
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_probe(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu"):
    """Train a probe and return train/test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # Evaluation
    def eval_accuracy(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    return eval_accuracy(train_loader), eval_accuracy(test_loader)


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, classification_report
)
import numpy as np

def evaluate_metrics(model, loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0)
    }
    return results