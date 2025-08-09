from daep.PhotometricLayers import photometricTransceiverEncoder2stages
from daep.util_layers import MLP
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split


import numpy as np

class PhotClassifier(nn.Module):
    def __init__(self, num_classes, num_bands = 1, 
                 bottleneck_length = 1,
                 bottleneck_dim = 64,
                 hidden_len = 128,
                 model_dim = 128, 
                 num_heads = 8, 
                 ff_dim = 128,
                 num_layers = 4,
                 dropout=0.1,
                 out_middle = [64],
                 selfattn=False, 
                 concat = True,
                 fourier = False):
        super().__init__()
        self.encoder = photometricTransceiverEncoder2stages(
            num_bands, 
                 bottleneck_length,
                 bottleneck_dim,
                 hidden_len ,
                 model_dim, 
                 num_heads, 
                 ff_dim,
                 num_layers,
                 dropout,
                 selfattn, 
                 concat,
                 fourier
            
        )
        self.classifier = MLP(bottleneck_dim, num_classes, out_middle)

    def forward(self, x):
        z = self.encoder(x)[:, 0, :]          # (batch, emb_dim)
        return self.classifier(z)     # (batch, num_classes)

def train(
                 bottleneck_dim = 128,
                 hidden_len = 32,
                 model_dim = 128, 
                 num_heads = 8, 
                 ff_dim = 256,
                 num_layers = 4,
                 
                 epochs = 50,
                 lr = 0.00025, 
                 dropout=0.1,
                 out_middle = [64],
                 selfattn = False, 
                 concat = True,
                 fourier = True):
    
    flux = np.load("../data/tess_variablestar/fluxes_train.npy")
    flux_err = np.load("../data/tess_variablestar/fluxes_errs_train.npy")
    flux_err[~np.isfinite(flux_err)] = 1e20
    time = np.load("../data/tess_variablestar/times_train.npy")
    label = np.load("../data/tess_variablestar/labels_train.npy", allow_pickle = True)[:, 0]
    classes, labels_int = np.unique(label, return_inverse=True)
    # classes: array(['cat', 'dog', 'mouse'], dtype='<U5')
    # labels_int: array([0, 1, 1, 0, 2, 0])

    # Convert to torch tensor
    labels_tensor = torch.tensor(labels_int, dtype=torch.long)
    
    mask = np.logical_or(~np.isfinite(flux), flux/flux_err <= 4.)
    mask = np.logical_or(mask, ~np.isfinite(time))
    
    time[mask] = 1e22
    time = time - np.min(time, axis = 1)[:, None]
    
    
    # per sample normalization, not ideal but if we do global things got super small
    flux = (flux - np.mean(flux, axis = 1, where = ~mask)[:, None])/(np.std(flux, axis = 1, where = ~mask)[:, None])
    flux[~np.isfinite(flux)] = 0. # constant cause trouble in the above line
    flux = np.clip(flux, -6, 6) # sanitize
    flux[mask] = 0.
    time[mask] = 0.
    
    n_label = labels_tensor.max() + 1
    #breakpoint()
    dataset = TensorDataset(torch.tensor(flux, dtype = torch.float32), 
                            torch.tensor(time, dtype = torch.float32), 
                            torch.tensor(mask, dtype = torch.float32), 
                            labels_tensor)
    train_size = int(0.8 * len(dataset))  # 80% train
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PhotClassifier(n_label, 1, 
                 1,
                 bottleneck_dim,
                 hidden_len,
                 model_dim, 
                 num_heads, 
                 ff_dim,
                 num_layers,
                 dropout,
                 out_middle,
                 selfattn, 
                 concat,
                 fourier).to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batchflux, batchtime, batchmask, batchlabel in train_loader:
            optimizer.zero_grad()
            logits = model({"flux": batchflux.to(device), 
                            "time": batchtime.to(device), 
                            "mask": batchmask.to(device)})
            loss = criterion(logits, batchlabel.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batchflux.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds.detach().cpu() ==  batchlabel).sum().item()
            total +=  batchlabel.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batchflux, batchtime, batchmask, batchlabel in val_loader:
                logits = model({"flux": batchflux.to(device), 
                            "time": batchtime.to(device), 
                            "mask": batchmask.to(device)})
                val_loss += criterion(logits, batchlabel.to(device)).item()
                preds = logits.argmax(dim=1).detach().cpu()
                correct += (preds.detach().cpu() == batchlabel).sum().item()
                total += batchlabel.size(0)
    
        val_loss /= len(val_loader)
        val_acc = correct / total
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )




import fire           

if __name__ == '__main__':
    fire.Fire(train)

