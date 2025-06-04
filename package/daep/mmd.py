import torch
import torch.nn as nn
import torchsde

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=1.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            #n_samples = L2_distances.shape[0]
            #return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
            with torch.no_grad():
                return torch.median(L2_distances)
        return self.bandwidth
    
    def get_bandwidth_from_data(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        #n_samples = L2_distances.shape[0]
        #return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        with torch.no_grad():
            return torch.median(L2_distances)

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / ((self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[:, None, None])).sum(dim=0)


class MMD(nn.Module):

    def __init__(self, kernel=RBF(bandwidth=1.)): # default to a fixed bandwidth for optimization
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
