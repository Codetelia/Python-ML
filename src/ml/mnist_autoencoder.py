"""
CNN-based Autoencoder for MNIST Dataset

This module implements a convolutional autoencoder for reconstructing MNIST images.
It includes data loading, model architecture, loss functions, and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional
import os


class CNNEncoder(nn.Module):
    """
    Convolutional encoder that compresses MNIST images into a latent representation.
    
    Architecture:
    - Conv2D (1 -> 32) + ReLU + MaxPool
    - Conv2D (32 -> 64) + ReLU + MaxPool
    - Conv2D (64 -> 128) + ReLU
    - Flatten + Linear to latent dimension
    """
    
    def __init__(self, latent_dim: int = 64):
        super(CNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 x 14 x 14
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 7 x 7
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 128 x 7 x 7
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CNNDecoder(nn.Module):
    """
    Convolutional decoder that reconstructs MNIST images from latent representation.
    
    Architecture:
    - Linear from latent dimension + Reshape
    - ConvTranspose2D (128 -> 64) + ReLU
    - ConvTranspose2D (64 -> 32) + ReLU
    - ConvTranspose2D (32 -> 1) + Sigmoid
    """
    
    def __init__(self, latent_dim: int = 64):
        super(CNNDecoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        
        self.decoder = nn.Sequential(
            # Input after reshape: 128 x 7 x 7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 64 x 14 x 14
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),  # 32 x 28 x 28
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 1 x 28 x 28, output in [0, 1]
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x


class CNNAutoencoder(nn.Module):
    """
    Complete CNN-based autoencoder combining encoder and decoder.
    """
    
    def __init__(self, latent_dim: int = 64):
        super(CNNAutoencoder, self).__init__()
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        """Encode images to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent vectors to images"""
        return self.decoder(z)


class MNISTDataLoader:
    """
    Data loader utility for MNIST dataset with train/test splits.
    """
    
    def __init__(self, batch_size: int = 128, data_dir: str = './data'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # Transform: convert to tensor and normalize to [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load MNIST train and test datasets.
        
        Returns:
            train_loader, test_loader
        """
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader


class AutoencoderLoss:
    """
    Loss functions for training the autoencoder.
    """
    
    @staticmethod
    def mse_loss(reconstruction, original):
        """Mean Squared Error loss"""
        return nn.functional.mse_loss(reconstruction, original)
    
    @staticmethod
    def bce_loss(reconstruction, original):
        """Binary Cross-Entropy loss"""
        return nn.functional.binary_cross_entropy(reconstruction, original)
    
    @staticmethod
    def combined_loss(reconstruction, original, alpha: float = 0.5):
        """
        Combined MSE and BCE loss.
        
        Args:
            alpha: weight for MSE (1-alpha for BCE)
        """
        mse = nn.functional.mse_loss(reconstruction, original)
        bce = nn.functional.binary_cross_entropy(reconstruction, original)
        return alpha * mse + (1 - alpha) * bce


class AutoencoderTrainer:
    """
    Trainer class for the CNN autoencoder with training loop and evaluation.
    """
    
    def __init__(
        self,
        model: CNNAutoencoder,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        loss_type: str = 'mse'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_type = loss_type
        
        # Select loss function
        if loss_type == 'mse':
            self.criterion = AutoencoderLoss.mse_loss
        elif loss_type == 'bce':
            self.criterion = AutoencoderLoss.bce_loss
        else:
            self.criterion = AutoencoderLoss.combined_loss
            
        self.train_losses = []
        self.test_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Returns:
            average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstruction = self.model(data)
            loss = self.criterion(reconstruction, data)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """
        Evaluate on test set.
        
        Returns:
            average test loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                reconstruction = self.model(data)
                loss = self.criterion(reconstruction, data)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(test_loader)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 20,
        verbose: bool = True
    ):
        """
        Complete training loop.
        
        Args:
            train_loader: training data loader
            test_loader: test data loader
            epochs: number of training epochs
            verbose: print progress
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            if verbose:
                print(f'Epoch {epoch}/{epochs} - '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Test Loss: {test_loss:.6f}')
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        print(f"Model loaded from {filepath}")


def visualize_reconstructions(
    model: CNNAutoencoder,
    test_loader: DataLoader,
    device: str,
    num_images: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize original images and their reconstructions.
    
    Args:
        model: trained autoencoder
        test_loader: test data loader
        device: computation device
        num_images: number of images to display
        save_path: path to save the figure (optional)
    """
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    data, _ = next(data_iter)
    data = data[:num_images].to(device)
    
    # Generate reconstructions
    with torch.no_grad():
        reconstructions = model(data)
    
    # Move to CPU for plotting
    data = data.cpu()
    reconstructions = reconstructions.cpu()
    
    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    
    for i in range(num_images):
        # Original images
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
            
        # Reconstructed images
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_training_curves(
    trainer: AutoencoderTrainer,
    save_path: Optional[str] = None
):
    """
    Plot training and test loss curves.
    
    Args:
        trainer: trainer object with loss history
        save_path: path to save the figure (optional)
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(trainer.train_losses) + 1)
    
    plt.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, trainer.test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def main():
    """
    Main function demonstrating usage of the autoencoder.
    """
    # Configuration
    BATCH_SIZE = 128
    LATENT_DIM = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print(f"Latent dimension: {LATENT_DIM}")
    print("-" * 50)
    
    # Load data
    print("Loading MNIST dataset...")
    data_loader = MNISTDataLoader(batch_size=BATCH_SIZE)
    train_loader, test_loader = data_loader.get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("-" * 50)
    
    # Create model
    print("Creating CNN Autoencoder...")
    model = CNNAutoencoder(latent_dim=LATENT_DIM)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 50)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = AutoencoderTrainer(
        model=model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        loss_type='mse'
    )
    
    # Train
    print(f"Training for {EPOCHS} epochs...")
    print("-" * 50)
    trainer.train(train_loader, test_loader, epochs=EPOCHS, verbose=True)
    print("-" * 50)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/mnist_autoencoder.pth')
    
    # Visualize results
    print("\nGenerating visualizations...")
    os.makedirs('results', exist_ok=True)
    visualize_reconstructions(
        model, test_loader, DEVICE, 
        num_images=10, 
        save_path='results/reconstructions.png'
    )
    plot_training_curves(trainer, save_path='results/training_curves.png')
    
    print("\nTraining complete!")
    print(f"Final train loss: {trainer.train_losses[-1]:.6f}")
    print(f"Final test loss: {trainer.test_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
