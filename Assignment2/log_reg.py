""" 
author:- Mahafujul Alam
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# For scikit-learn comparisons and clustering:
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
batch_size = 68
num_epochs = 30
learning_rates = {"SGD": 0.5, "Adam": 0.001, "RMSprop": 0.001}
lambda_reg = 0.001  # L2 regularization factor

# Load Fashion-MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset_full = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True)

# Split into training and validation sets (90% train, 10% validation)
val_split = 0.1
num_val = int(len(train_dataset_full) * val_split)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset_full, [len(train_dataset_full) - num_val, num_val])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# Plot a few sample images from a dataset.


def plot_sample_images(dataset, num_images=9):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    indices = np.random.choice(len(dataset), num_images, replace=False)
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.squeeze().numpy()
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Plot the model's weight matrices (each weight vector reshaped as a 28x28 image)


def plot_model_weights(model):
    weights = model.linear.weight.data.cpu().numpy()  # shape: (10, 784)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < weights.shape[0]:
            weight_img = weights[i].reshape(28, 28)
            ax.imshow(weight_img, cmap="viridis")
            ax.set_title(f"Class {i}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


# Basic Logistic Regression Model


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  # 28x28 input, 10 output classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten images
        return torch.softmax(self.linear(x), dim=1)

# Logistic Regression Model with Dropout for additional regularization.


class LogisticRegressionDropoutModel(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(LogisticRegressionDropoutModel, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dropout(x)
        return torch.softmax(self.linear(x), dim=1)


# Train model and record history along with per-epoch duration.

def train_model(optimizer_name, learning_rate, num_epochs, lambda_reg, use_dropout=False):
    if use_dropout:
        model = LogisticRegressionDropoutModel().to(device)
    else:
        model = LogisticRegressionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, weight_decay=lambda_reg)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=lambda_reg)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=lambda_reg)

    history = {"train_loss": [], "val_loss": [],
               "train_accuracy": [], "val_accuracy": [], "epoch_time": []}

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        epoch_duration = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time"].append(epoch_duration)

        print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}, Time = {epoch_duration:.2f}s")
    return model, history


history_dict = {}
models = {}

print("\n--- Training without Dropout ---")
for optimizer_name, lr in learning_rates.items():
    print(
        f"\nTraining with {optimizer_name} optimizer (lr={lr}, lambda_reg={lambda_reg})")
    model, history = train_model(
        optimizer_name, lr, num_epochs, lambda_reg, use_dropout=False)
    history_dict[optimizer_name] = history
    models[optimizer_name] = model

# Plot loss and accuracy curves for each optimizer.
epochs = range(1, num_epochs + 1)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
for opt_name, history in history_dict.items():
    axs[0].plot(epochs, history["train_loss"], label=f"{opt_name} Train")
    axs[0].plot(epochs, history["val_loss"], '--', label=f"{opt_name} Val")
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

for opt_name, history in history_dict.items():
    axs[1].plot(epochs, history["train_accuracy"], label=f"{opt_name} Train")
    axs[1].plot(epochs, history["val_accuracy"], '--', label=f"{opt_name} Val")
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


print("\n--- Optimizer Performance Summary ---")
for opt_name, history in history_dict.items():
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val_acc = max(history["val_accuracy"])
    avg_epoch_time = np.mean(history["epoch_time"])
    print(f"{opt_name}: Best Epoch = {best_epoch}, Best Val Acc = {best_val_acc:.4f}, Avg Epoch Time = {avg_epoch_time:.2f}s")

# Overfitting Analysis: Plot the gap between training and validation accuracy.
plt.figure(figsize=(8, 6))
for opt_name, history in history_dict.items():
    gap = np.array(history["train_accuracy"]) - \
        np.array(history["val_accuracy"])
    plt.plot(epochs, gap, label=f"{opt_name} Gap")
plt.title("Train-Validation Accuracy Gap over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy Gap")
plt.legend()
plt.grid(True)
plt.show()


print("\n--- Random Forest and SVM Comparison ---")
# Function to convert a dataset to numpy arrays (subsample for speed)


def convert_dataset_to_numpy(dataset, num_samples):
    X = []
    y = []
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        image, label = dataset[idx]
        X.append(image.numpy().reshape(-1))
        y.append(label)
    return np.array(X), np.array(y)


# Use 10000 samples for training and 2000 for testing.
X_train_np, y_train_np = convert_dataset_to_numpy(train_dataset_full, 10000)
X_test_np, y_test_np = convert_dataset_to_numpy(test_dataset, 2000)

# Train and evaluate Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_np, y_train_np)
rf_preds = rf_clf.predict(X_test_np)
rf_acc = accuracy_score(y_test_np, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Train and evaluate SVM (linear kernel for speed)
svm_clf = SVC(kernel="linear", random_state=42)
svm_clf.fit(X_train_np, y_train_np)
svm_preds = svm_clf.predict(X_test_np)
svm_acc = accuracy_score(y_test_np, svm_preds)
print(f"SVM Accuracy: {svm_acc:.4f}")


print("\n--- Clustering and Visualization of Model Weights ---")
# Use one trained model, e.g., the model trained with Adam.
selected_model = models["Adam"]
weights = selected_model.linear.weight.data.cpu().numpy()  # shape: (10, 784)

# Use t-SNE to reduce the weight dimensions to 2D.
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
weights_2d = tsne.fit_transform(weights)

# Apply k-means clustering (for demonstration, choose 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(weights)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    weights_2d[:, 0], weights_2d[:, 1], c=clusters, cmap="viridis", s=100)
for i in range(weights.shape[0]):
    plt.annotate(
        f"Class {i}", (weights_2d[i, 0], weights_2d[i, 1]), fontsize=12)
plt.title("t-SNE Visualization of Model Weights with K-Means Clustering")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# Additionally, visualize the weight matrices as images.
print("\n--- Visualizing Model Weights as Images ---")
plot_model_weights(selected_model)

print("\n--- Sample Training Images ---")
plot_sample_images(train_dataset_full)
