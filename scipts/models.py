# Imports as always...
import torch.nn as nn
import torch.nn.functional as F
from icoCNN.icoCNN import ConvIco, PoolIco, LNormIco


class SphericalCNNClassifier(nn.Module):
    """
    Spherical Convolutional Neural Network (Spherical CNN) for classification.

    Args:
        -
    """
    # TODO: Fucking all of it.


class IcoCNNClassifier(nn.Module):
    """
    Icosahedral Convolutional Neural Network (IcoMNIST) for classification.

    Args:
        - r (int): Resolution of the icosahedral grid.
        - in_channels (int): Number of channels in the input (icosahedral) signal, excluding orientation channels.
        - out_channels (int): Number of outcome classes.
        - R_in (int): 1 for scalar, 6 for regular feature vectors.
        - bias (bool): Whether to use bias in convolutional layers.
        - smooth_vertices (bool): Whether to smooth the vertices of the icosahedral grid.
    """

    def __init__(self, r, in_channels, out_channels, R_in=1, bias=True, smooth_vertices=False):
        super(IcoCNNClassifier, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R_in = R_in
        self.bias = bias

        # Convolution layers.
        self.conv1 = ConvIco(r, Cin=in_channels, Cout=16, Rin=R_in, Rout=6, bias=bias, smooth_vertices=smooth_vertices)
        self.conv2 = ConvIco(r-1, Cin=16, Cout=32, Rin=6, Rout=1, bias=bias, smooth_vertices=smooth_vertices)

        # Pooling layer.
        # Shapes as [B, C, 5, 2^r, 2^(r+1)] -> [B, C, 5, 2^(r-1), 2^r].
        self.pool1 = PoolIco(r, R=6, smooth_vertices=smooth_vertices)
        self.pool2 = PoolIco(r-1, R=1, smooth_vertices=smooth_vertices)

        # TODO: Batch normalisation.

        # Fully connected linear layer.
        self.fc = nn.Linear(32 * 5 * 2**(r-2) * 2**(r-1), out_channels, bias=bias)

    def forward(self, x):
        # Shape going in: [B, in_channels, R_in, 5, 2^r, 2^(r+1)].

        # --- Convolutions ---

        # [B, in_channels, R_in, 5, 2^r, 2^(r+1)] -> [B, 16, 6, 5, 2^(r-1), 2^r].
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # [B, 16, 6, 5, 2^(r-1), 2^r] -> [B, 32, 6, 5, 2^(r-2), 2^(r-1)].
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # --- Flatten ---

        # [B, 64, 1, 5, 2^(r-2), 2^(r-1)] -> [B, 64 * 5 * 2^(r-2) * 2^(r-1)].
        x = x.view(x.size(0), -1)

        # --- Fully connected ---

        x = self.fc(x)

        # Shape going out: [B, out_channels].

        return x


class IcoCNNClassifierNoPooling(nn.Module):
    """
    Icosahedral Convolutional Neural Network (IcoMNIST) for classification.

    Args:
        - r (int): Resolution of the icosahedral grid.
        - in_channels (int): Number of channels in the input (icosahedral) signal, excluding orientation channels.
        - out_channels (int): Number of outcome classes.
        - R_in (int): 1 for scalar, 6 for regular feature vectors.
        - bias (bool): Whether to use bias in convolutional layers.
        - smooth_vertices (bool): Whether to smooth the vertices of the icosahedral grid.
    """

    def __init__(self, r, in_channels, out_channels, R_in=1, bias=True, smooth_vertices=False):
        super(IcoCNNClassifierNoPooling, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R_in = R_in
        self.bias = bias

        # Convolution layers.
        self.conv1 = ConvIco(r, Cin=in_channels, Cout=16, Rin=R_in, Rout=6, bias=bias, smooth_vertices=smooth_vertices)
        self.conv2 = ConvIco(r, Cin=16, Cout=32, Rin=6, Rout=1, bias=bias, smooth_vertices=smooth_vertices)

        # TODO: Batch normalisation.

        # Fully connected linear layer.
        self.fc = nn.Linear(32 * 5 * 2**r * 2**(r+1), out_channels, bias=bias)

    def forward(self, x):
        # Shape going in: [B, in_channels, R_in, 5, 2^r, 2^(r+1)].

        # --- Convolutions ---

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # --- Flatten ---

        x = x.view(x.size(0), -1)

        # --- Fully connected ---

        x = self.fc(x)

        # Shape going out: [B, out_channels].

        return x
