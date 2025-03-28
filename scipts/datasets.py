# Imports as always...
import numpy as np

import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

from scipy.spatial import cKDTree

from icoCNN.tools import random_icosahedral_rotation_matrix, rotate_signal


class SphericalMNIST(Dataset):
    """
    Online version of Spherical MNIST -- projected to sphere on call.
    Not an efficient implementation, it must be said; not to be used in training.
    """

    def __init__(self, train=True):
        # Transforms.
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
        ])

        # The standard MNIST data.
        self.mnist_data = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # Selecgt the image and corresponding label.
        image, label = self.mnist_data[idx]

        # Convert to normalised numpy array.
        image = image.squeeze().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        # Define spherical coordinates
        rows, cols = image.shape
        theta = np.linspace(0, np.pi, rows)  # Polar angle
        phi = np.linspace(0, 2 * np.pi, cols)  # Azimuthal angle
        theta, phi = np.meshgrid(theta, phi)

        # Convert spherical to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Return 3D coordinates, image intensity, and label
        return torch.tensor(x), torch.tensor(y), torch.tensor(z), torch.tensor(image), torch.tensor(label)


class IcosahedralMNIST(Dataset):
    def __init__(self, coord_tensor, train=True, augment=False):
        """
        Online version of Icosahedral MNIST -- projected to sphere, then icosahedron on call.

        Args:
            - coord_tensor (torch.Tensor): 3D co-ordinates of the icosahedron's vertices. Shape [5, 2**r, 2**(r+1), 3] where the last dimension is (x, y, z).
            - train (bool): Whether to load the MNIST training set.
            - augment (bool): Whether to augment with random icosahedral symmetries.
        """

        # Keep relative to given co-ordiantes.
        assert coord_tensor.shape[-1] == 3, "Last dimension must be (x, y, z) coordinates."
        self.coord_tensor = coord_tensor

        # Load MNIST dataset.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_data = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

        # Augmentation.
        self.augment = augment

    def __len__(self):
        return len(self.mnist_data)

    def ico_symmetry_augment(self, ico_signal):
        """
        Augments the given icosahedral signal by performing a random icosahedral rotation.

        Args:
            ico_signal (torch.Tensor): Shape [1, 1, 5, 2**r, 2**(r+1)].
        """

        # Sample a random rotation matrix from the 60 icosahedral symmetries.
        rotation_matrix = random_icosahedral_rotation_matrix()

        # Apply the rotation to the signal.
        rotated_signal = rotate_signal(ico_signal, rotation_matrix, self.coord_tensor.detach().cpu().numpy())

        return rotated_signal

    def __getitem__(self, idx):
        """
        Maps each MNIST pixel to its nearest 3D coordinate.

        Args:
            idx (int): Index of the MNIST digit.
        """
        image, label = self.mnist_data[idx]
        image = image.squeeze().numpy()  # Convert to 28x28 numpy array.
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0,1].

        # Generate spherical coordinates for MNIST image (28x28).
        rows, cols = image.shape
        theta = np.linspace(0, np.pi, rows)  # Polar angle (from top to bottom of the sphere).
        phi = np.linspace(0, 2 * np.pi, cols)  # Azimuthal angle (around the sphere).
        theta, phi = np.meshgrid(theta, phi)

        # Convert to Cartesian coordinates (x, y, z).
        x_proj = np.sin(theta) * np.cos(phi)
        y_proj = np.sin(theta) * np.sin(phi)
        z_proj = np.cos(theta)

        # Flatten for nearest neighbor search.
        proj_coords = np.vstack([x_proj.ravel(), y_proj.ravel(), z_proj.ravel()]).T
        image_values = image.ravel()

        # KD-tree for nearest neighbor lookup.
        tree = cKDTree(proj_coords)

        # Reshape input coordinates and find nearest MNIST-pixel projection.
        input_coords = self.coord_tensor.reshape(-1, 3).numpy()  # Shape: [num_points, 3].
        _, nearest_indices = tree.query(input_coords)
        interpolated_values = image_values[nearest_indices]

        # Reshape to original input shape [5, 2**r, 2**(r+1)].
        output_tensor = interpolated_values.reshape(self.coord_tensor.shape[:-1])
        output_tensor = torch.tensor(output_tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 2**r, 2**(r+1)].

        if self.augment:
            output_tensor = self.ico_symmetry_augment(output_tensor)

        return output_tensor, label


class PrecomputedIcosahedralMNIST(Dataset):
    def __init__(self, r, augment='none'):
        """
        Pre-computed Icosahedral MNIST dataset. Reads data in locally.

        Args:
            - r (int): Resolution of the icosahedral grid.
            - augment (str): Type of augmentation to use; choosing from
                - 'none': No augmentation.
                - 'ico': A single (random) icosahedral rotation on each datum.
                - 'all ico': All 60 icosahedral symmetries on each datum.
                - 'rot': A single (random) rotation in SO(3) on each datum (projection onto icosahedron).
                - 'all rot': 60 (randomly sampled) rotations in SO(3) on each datum (projection onto icosahedron).
        """

        # Load projected MNIST dataset (with augmentation).
        if augment == 'ico':
            print('Random Icosahedral Rotation Augmentation.')
            self.projected_mnist_data = torch.load(f'./data/IcoMNIST/ico_projected_mnist_augmented_(r={r}).pt')
        elif augment == 'all ico':
            # For reference, for r=2, this takes ~30s (the file is ~2.2 GB).
            print('All Icosahedral Symmetries Augmentation.')
            self.projected_mnist_data = torch.load(f'./data/IcoMNIST/ico_projected_mnist_all_symmetries_(r={r}).pt')
        else:
            print('No Augmentation.')
            self.projected_mnist_data = torch.load(f'./data/IcoMNIST/ico_projected_mnist_(r={r}).pt')



    def __len__(self):
        return len(self.projected_mnist_data['data'])

    def __getitem__(self, idx):
        """
        Maps each MNIST pixel to its nearest 3D coordinate.

        Args:
            idx (int): Index of the MNIST digit.
        """
        output_tensor = self.projected_mnist_data['data'][idx]
        label = self.projected_mnist_data['labels'][idx]

        return output_tensor, label
