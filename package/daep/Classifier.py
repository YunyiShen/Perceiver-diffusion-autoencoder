import torch
import torch.nn as nn
import torch.nn.functional as F
from daep.PhotometricLayers import photometricTransceiverEncoder2stages
from daep.util_layers import MLP

class LCC(nn.Module):
    """
    Linear Classification Classifier (LCC) for classification tasks.
    
    A feedforward neural network classifier with CNN feature extraction that takes 
    encoded bottleneck representations and outputs class probabilities.
    """
    def __init__(self, bottleneck_dim, bottleneck_len, dropout_p, num_classes):
        super(LCC, self).__init__()
        
        final_outchannels = 8
        # CNN layers for feature extraction from bottleneck representation
        # Input: (batch_size, 1, bottleneck_len, bottleneck_dim)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(in_channels=4, out_channels=final_outchannels, kernel_size=3, padding=1)
        self.cnn_activation = nn.SiLU()
        self.cnn_pool = nn.AdaptiveAvgPool2d((bottleneck_len, bottleneck_dim))
        
        # Calculate total CNN output dimension after flattening
        # CNN output: (batch_size, 16, bottleneck_len, bottleneck_dim)
        # Flattened: (batch_size, 16 * bottleneck_len * bottleneck_dim)
        self.total_cnn_output_dim = final_outchannels * bottleneck_len * bottleneck_dim
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.total_cnn_output_dim, self.total_cnn_output_dim)
        # self.fc2 = nn.Linear(self.total_cnn_output_dim, self.total_cnn_output_dim)
        self.fc3 = nn.Linear(self.total_cnn_output_dim, 20)
        self.fc4 = nn.Linear(20, num_classes)
        
        # Activation and normalization layers
        self.swish = nn.SiLU()
        # Fixed: Use correct dimensions for layer normalization after CNN processing
        self.norm0 = nn.LayerNorm(normalized_shape=self.total_cnn_output_dim)
        # self.norm1 = nn.LayerNorm(normalized_shape=self.total_cnn_output_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=20)
        self.norm3 = nn.LayerNorm(normalized_shape=num_classes)
        
        # Dropout layers for regularization
        self.dropout = nn.Dropout(dropout_p)
        self.dropout0 = nn.Dropout(dropout_p)

    def forward(self, x, t=None, mask=None):
        """
        Forward pass through the classifier.
        
        Parameters
        ----------
        x : torch.Tensor
            Input bottleneck representation tensor
            Shape: (batch_size, bottleneck_len, bottleneck_dim) or (batch_size, bottleneck_dim * bottleneck_len)
        t : torch.Tensor, optional
            Time parameter (not used in computation, kept for compatibility)
        mask : torch.Tensor, optional
            Mask tensor (not used in computation, kept for compatibility)
            
        Returns
        -------
        torch.Tensor
            Class probabilities with shape (batch_size, num_classes)
        """
        # Ensure input has proper dimensions for CNN processing
        if x.dim() == 3:
            # x: (batch_size, bottleneck_len, bottleneck_dim)
            x_cnn = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, bottleneck_len, bottleneck_dim)
        else:
            raise ValueError(f"Input tensor must be 3D for CNN processing, got {x.dim()}D tensor with shape {x.shape}")

        # CNN feature extraction
        x_cnn = self.cnn1(x_cnn)
        x_cnn = self.cnn_activation(x_cnn)
        x_cnn = self.cnn2(x_cnn)
        x_cnn = self.cnn_activation(x_cnn)
        x_cnn = self.cnn_pool(x_cnn)  # (batch_size, 16, bottleneck_len, bottleneck_dim)

        # Flatten CNN output for fully connected layers
        x_flat = x_cnn.view(x_cnn.size(0), -1)  # (batch_size, 16 * bottleneck_len * bottleneck_dim)

        # Apply dropout and first fully connected layer
        x = self.dropout(x_flat)
        x = self.fc1(x)
        x = self.norm0(x)  # Fixed: Now uses correct dimension
        x = self.swish(x)
        x = self.dropout0(x)

        # # Second fully connected layer
        # x = self.fc2(x)
        # x = self.norm1(x)  # Fixed: Now uses correct dimension
        # x = self.swish(x)
        # x = self.dropout0(x)

        # Third fully connected layer
        x = self.fc3(x)
        x = self.norm2(x)
        x = self.swish(x)
        x = self.dropout0(x)

        # Final classification layer
        x = self.fc4(x)
        x = self.norm3(x)

        # Apply softmax to get class probabilities
        # Ensure output shape is always (batch_size, num_classes)
        x = F.softmax(x, dim=-1)

        return x


class PhotClassifier(nn.Module):
    def __init__(self, num_classes, num_bands = 1, 
                 bottleneck_length = 1,
                 bottleneck_dim = 128,
                 hidden_len = 32,
                 model_dim = 128, 
                 num_heads = 8, 
                 ff_dim = 256,
                 num_layers = 4,
                 dropout = 0.1,
                 out_middle = [64],
                 selfattn = False, 
                 concat = True,
                 fourier = True):
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