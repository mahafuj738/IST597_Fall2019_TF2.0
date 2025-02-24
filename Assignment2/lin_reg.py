"""
author:Mahafujul Alam
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import hashlib


# Setting up a unique seed for reproducibility
def name_to_seed(name):
    # Hash the name using SHA256
    hash_object = hashlib.sha256(name.encode('utf-8'))
    # Convert the hexadecimal digest to an integer
    seed = int(hash_object.hexdigest(), 16)
    # Limit the seed to a 32-bit integer range
    seed = seed % (2**32)
    return seed


seed = name_to_seed('alammm')
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

N = 10000  # Number of data points
x = torch.rand(N, 1) * 10  # Random x values between 0 and 10


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(
            1, requires_grad=True))  # Random init
        self.b = nn.Parameter(torch.randn(
            1, requires_grad=True))  # Random init

    def forward(self, x):
        return self.W * x + self.b

# Loss functions


def mse_loss(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)  # L2 Loss


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_true - y_pred))  # L1 Loss


def hybrid_loss(y_pred, y_true, alpha=0.5):
    return alpha * mse_loss(y_pred, y_true) + (1 - alpha) * mae_loss(y_pred, y_true)

# Noise Injection Functions


def add_noise(tensor, noise_type="gaussian", std=.0):
    if noise_type == "gaussian":
        return tensor + torch.randn_like(tensor) * std
    elif noise_type == "uniform":
        return tensor + (torch.rand_like(tensor) - 0.5) * std * 2
    elif noise_type == "laplacian":
        return tensor + torch.from_numpy(np.random.laplace(0, std * 0.5, tensor.shape).astype(np.float32))
    return tensor


noise_std = 0.5
# Our true function y = 3x + 2 + noise
y = 3 * x + 2 + add_noise(torch.zeros_like(x), "gaussian", noise_std)


def train(loss_fn, initial_lr=0.001, epochs=2500, patience=300, noise_type="gaussian", add_noise_to_data=False, add_noise_to_weights=False, add_noise_to_lr=False):
    model = LinearRegression()
    # Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    best_loss = float('inf')
    patience_counter = 0  # Track epochs with no improvement
    loss_history = []  # Store loss per epoch for plotting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    x_train, y_train = x.to(device), y.to(device)

    start_time = time.time()  # Track time

    for epoch in range(epochs):
        optimizer.zero_grad()

        if add_noise_to_data:
            x_train_noisy = add_noise(x_train, noise_type, noise_std)
            y_train_noisy = add_noise(y_train, noise_type, noise_std)
        else:
            x_train_noisy, y_train_noisy = x_train, y_train

        y_pred = model(x_train_noisy)

        if add_noise_to_weights:
            model.W.data += add_noise(torch.zeros_like(model.W),
                                      noise_type, 0.01)
            model.b.data += add_noise(torch.zeros_like(model.b),
                                      noise_type, 0.01)

        if add_noise_to_lr:
            for param_group in optimizer.param_groups:
                # param_group["lr"] *= (1 + torch.clamp(add_noise(torch.zeros(1), noise_type, 0.005).item(), -0.02, 0.02))
                param_group["lr"] *= (1 + float(torch.clamp(
                    add_noise(torch.zeros(1), noise_type, 0.005), min=-0.02, max=0.02).item()))

        loss = loss_fn(y_pred, y_train_noisy)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())  # Store loss

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(optimizer.param_groups[0]["lr"])
                optimizer.param_groups[0]["lr"] *= 0.5  # Reduce learning rate
                print(
                    f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.6f} at step {epoch}")
                patience_counter = 0

        # if epoch % 10 == 0 or epoch == epochs - 1:
        current_lr = optimizer.param_groups[0]['lr']
        # print(f'current_lr: {current_lr}')

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:03d}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}, W: {model.W.item():.4f}, b: {model.b.item():.4f}")

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.3f} sec on {device}")
    print(
        f"\nFinal Model: W = {model.W.item():.4f}, b = {model.b.item():.4f}, Final Loss: {loss.item():.4f}")

    # Plot loss curve
    plt.plot(loss_history, label=f'{loss_fn.__name__}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

    return model  # Return trained model


print("\nTraining with Gaussian Noise in Data...")
model_data_noisy = train(mse_loss, initial_lr=0.001,
                         epochs=30000, noise_type="gaussian")

# print("\nTraining with Gaussian Noise in Data...")
# model_data_noisy = train(mse_loss, initial_lr=0.001, epochs=30000, noise_type="gaussian", add_noise_to_data=True, add_noise_to_weights=True, add_noise_to_lr=True)

# print("\nTraining with Uniform Noise in Weights...")
# model_weight_noisy = train(mse_loss, initial_lr=0.1, epochs=1000, noise_type="uniform", add_noise_to_weights=True)

# print("\nTraining with Laplacian Noise in Learning Rate...")
# model_lr_noisy = train(mse_loss, initial_lr=0.1, epochs=1000, noise_type="laplacian", add_noise_to_lr=True)
