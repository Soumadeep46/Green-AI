import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torchvision import transforms
from unet_model import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


class SaplingDataset(Dataset):
    def _init_(self, images, labels, transform=None):
        # Ensure images are in the correct format (N, C, H, W)
        if images.shape[1] != 3:  # If channels are not in the correct position
            images = np.transpose(images, (0, 3, 1, 2))

        self.images = torch.FloatTensor(images)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def _getitem_(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _len_(self):
        return len(self.images)


def load_data(data_path):
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    all_data = []

    for file in data_files:
        try:
            data = np.load(os.path.join(data_path, file))
            if len(data.shape) == 4:  # Ensure correct shape (batch, height, width, channels)
                all_data.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not all_data:
        return None

    data = np.concatenate(all_data, axis=0)
    print(f"Loaded data shape: {data.shape}")
    return data


def create_dummy_labels(data):
    num_samples = data.shape[0]
    height = data.shape[2]  # Height is the third dimension after transpose
    width = data.shape[3]  # Width is the fourth dimension after transpose

    # Create binary masks for each image
    labels = np.zeros((num_samples, 1, height, width), dtype=np.float32)

    # Add some random sapling locations
    for i in range(num_samples):
        # Create 5-10 random sapling locations per image
        num_saplings = np.random.randint(5, 11)
        for _ in range(num_saplings):
            # Random position
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            # Create a small circular mask for each sapling
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if dx * dx + dy * dy <= 25:  # Circular mask of radius 5
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            labels[i, 0, ny, nx] = 1.0

    return labels


class FocalLoss(nn.Module):
    def _init_(self, alpha=0.25, gamma=2):
        super(FocalLoss, self)._init_()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


def train_model(data_path, model_path):
    print("Loading data...")
    X = load_data(data_path)
    if X is None:
        print("No data found!")
        return

    print(f"Original data shape: {X.shape}")

    # Convert to PyTorch format (N, C, H, W) before any other processing
    X = np.transpose(X, (0, 3, 1, 2))
    print(f"Transposed data shape: {X.shape}")

    # Normalize data
    X = (X - X.mean(axis=(0, 2, 3), keepdims=True)) / (X.std(axis=(0, 2, 3), keepdims=True) + 1e-8)

    # Create labels with matching dimensions
    y = create_dummy_labels(X)
    print(f"Labels shape: {y.shape}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = SaplingDataset(X_train, y_train)
    val_dataset = SaplingDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    num_epochs = 1
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Debug print for shapes
            if batch_idx == 0:
                print(f"Batch shapes - Input: {data.shape}, Target: {target.shape}")

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        print(f'Epoch {epoch}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed")


if __name__ == "_main_":
    train_model("processed_data", "sapling_detection_model_v3.pth")
