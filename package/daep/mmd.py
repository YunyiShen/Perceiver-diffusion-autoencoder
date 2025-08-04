import torch
import torch.nn as nn

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


def robust_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: torch.Tensor,
    labels_err: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Robust mean squared error loss that accounts for both predicted and input uncertainties.
    
    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values.
    y_pred : torch.Tensor
        Predicted values.
    variance : torch.Tensor
        Predicted log-variance from the model.
    labels_err : torch.Tensor
        Input uncertainty values.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-8.
        
    Returns
    -------
    torch.Tensor
        Robust MSE loss value.
    """
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    total_var = torch.exp(variance) + torch.square(labels_err) + epsilon
    wrapper_output = 0.5 * (
        (torch.square(y_true - y_pred) / total_var) + torch.log(total_var)
    )

    losses = wrapper_output.sum() / y_true.shape[0]
    return losses


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Standard mean squared error loss.
    
    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth values.
    y_pred : torch.Tensor
        Predicted values.
        
    Returns
    -------
    torch.Tensor
        MSE loss value.
    """
    losses = (torch.square(y_true - y_pred)).sum() / y_true.shape[0]
    return losses
