import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def create_permuted_mnist_vectorized(permutation, train=True):
    """
    Loads MNIST and applies the given permutation to the flattened images.
    Each image is reshaped to a 784-dimensional vector and then re-ordered.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Flatten the image to 784 dimensions
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = datasets.MNIST(root='./data', train=train,
                             download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(loader))
    data = data[:, permutation]  # Apply the permutation to all images
    return TensorDataset(data, targets)


class MLP(nn.Module):
    def __init__(self, depth=2, dropout_prob=0.5):
        """
        depth: total number of layers (input + hidden(s) + output).
               For example, depth=2 means no hidden layer (input -> output),
               depth=3 means one hidden layer.
        dropout_prob: probability for dropout.
        """
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(784, 256)
        self.dropout = nn.Dropout(dropout_prob)
        # Create hidden layers (depth-2 hidden layers if depth > 2)
        hidden_layers = []
        for _ in range(depth - 2):
            hidden_layers.append(nn.Linear(256, 256))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        logits = self.output_layer(x)
        return logits


def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]


def compute_loss(logits, labels, loss_type="NLL"):
    if loss_type == "NLL":
        loss_fn = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLL loss
        return loss_fn(logits, labels)
    elif loss_type == "L1":
        softmax_logits = F.softmax(logits, dim=1)
        target_one_hot = one_hot(labels, num_classes=10).to(logits.device)
        loss = F.l1_loss(softmax_logits, target_one_hot)
        return loss
    elif loss_type == "L2":
        softmax_logits = F.softmax(logits, dim=1)
        target_one_hot = one_hot(labels, num_classes=10).to(logits.device)
        loss = F.mse_loss(softmax_logits, target_one_hot)
        return loss
    elif loss_type == "L1+L2":
        softmax_logits = F.softmax(logits, dim=1)
        target_one_hot = one_hot(labels, num_classes=10).to(logits.device)
        loss_l1 = F.l1_loss(softmax_logits, target_one_hot)
        loss_l2 = F.mse_loss(softmax_logits, target_one_hot)
        return loss_l1 + loss_l2
    else:
        raise ValueError("Unsupported loss type")


def train_model(model, train_loader, epochs, loss_type, optimizer, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(
                device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = compute_loss(logits, batch_labels, loss_type)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_data.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_data.size(0)
        epoch_loss /= total
        accuracy = correct / total
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return model


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += data.size(0)
    accuracy = correct / total
    return accuracy


def get_optimizer(model, optimizer_type, lr=0.001):
    if optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer type")


def run_sequential_training(model, task_train_loaders, task_test_loaders, loss_type, optimizer_type, device, epochs_first=50, epochs_other=20):
    """
    Trains the model sequentially on each task.
    After training on a task, the model is evaluated on all tasks seen so far.
    Returns a performance matrix R where R[t, j] is the accuracy on task j after training on task t.
    """
    num_tasks = len(task_train_loaders)
    R = np.zeros((num_tasks, num_tasks))
    for t in range(num_tasks):
        print(f"\n=== Training on Task {t+1}/{num_tasks} ===")
        epochs = epochs_first if t == 0 else epochs_other
        optimizer = get_optimizer(model, optimizer_type, lr=0.001)
        model = train_model(
            model, task_train_loaders[t], epochs, loss_type, optimizer, device)
        # Evaluate on all tasks seen so far
        for test_t in range(t+1):
            acc = test_model(model, task_test_loaders[test_t], device)
            R[t, test_t] = acc
            print(f"--> Evaluation on Task {test_t}: Accuracy = {acc:.4f}")
    return R


def calculate_ACC(R):
    """Calculate average accuracy over tasks (diagonal of matrix R)."""
    return np.mean(np.diag(R))


def calculate_BWT(R):
    """
    Calculate Backward Transfer (BWT) as the average over all tasks (i>j) of:
      (R[i, j] - R[j, j])
    """
    num_tasks = R.shape[0]
    bwt = 0
    count = 0
    for i in range(1, num_tasks):
        for j in range(i):
            bwt += R[i, j] - R[j, j]
            count += 1
    return bwt / count if count > 0 else 0


def calculate_CBWT(R):
    """
    For each task t (except the final one), compute:
    CBWT(t) = 1/(T-t-1) * sum_{i=t+1}^{T-1} (R[i,t] - R[t,t])
    Returns both the list of CBWT for each task and their average.
    """
    T = R.shape[0]
    cbwt_values = []
    for t in range(T-1):
        diff_sum = 0
        count = T - t
        for i in range(t+1, T):
            diff_sum += (R[i, t] - R[t, t])
        cbwt_t = diff_sum / count
        cbwt_values.append(cbwt_t)
    avg_cbwt = np.mean(cbwt_values) if cbwt_values else 0
    return cbwt_values, avg_cbwt


def calculate_TBWT(R):
    """
    Compute TBWT using:
      TBWT = 1/(T-1) * sum_{i=1}^{T-1} (R[T-1, i] - R[i, i])
    where T-1 is the index of the final task.
    """
    T = R.shape[0]
    diff_sum = 0
    for i in range(1, T):
        diff_sum += (R[T-1, i] - R[i, i])
    tbwt = diff_sum / (T-1) if T > 1 else 0
    return tbwt


def run_experiments(device):
    # Define hyperparameter ranges
    loss_types = ["NLL", "L1", "L2", "L1+L2"]
    # optimizer_types = ["SGD", "Adam", "RMSProp"]
    optimizer_types = ["SGD", "Adam"]
    depths = [2, 3, 4]
    dropout_rates = [0.2, 0.6]

    results = []
    num_tasks_to_run = 10
    batch_size = 10000

    for loss_type in loss_types:
        for optimizer_type in optimizer_types:
            for depth in depths:
                for dropout_prob in dropout_rates:
                    print(
                        f"\n=== Experiment: loss={loss_type}, optimizer={optimizer_type}, depth={depth}, dropout={dropout_prob} ===")
                    # Initialize model with current configuration
                    model = MLP(
                        depth=depth, dropout_prob=dropout_prob).to(device)

                    # Create task-specific data loaders
                    task_train_loaders = []
                    task_test_loaders = []

                    # Task 0: Original MNIST using identity permutation
                    perm_identity = np.arange(784)
                    train_dataset = create_permuted_mnist_vectorized(
                        perm_identity, train=True)
                    test_dataset = create_permuted_mnist_vectorized(
                        perm_identity, train=False)
                    task_train_loaders.append(DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True))
                    task_test_loaders.append(DataLoader(
                        test_dataset, batch_size=batch_size, shuffle=False))

                    # Additional tasks: Random permutations
                    for k in range(1, num_tasks_to_run):
                        perm = np.random.permutation(784)
                        train_dataset = create_permuted_mnist_vectorized(
                            perm, train=True)
                        test_dataset = create_permuted_mnist_vectorized(
                            perm, train=False)
                        task_train_loaders.append(DataLoader(
                            train_dataset, batch_size=batch_size, shuffle=True))
                        task_test_loaders.append(DataLoader(
                            test_dataset, batch_size=batch_size, shuffle=False))

                    # Run sequential training over tasks and get performance matrix R
                    R = run_sequential_training(model, task_train_loaders, task_test_loaders,
                                                loss_type, optimizer_type, device,
                                                epochs_first=50, epochs_other=20)
                    ACC = calculate_ACC(R)
                    BWT = calculate_BWT(R)
                    _, avg_CBWT = calculate_CBWT(R)
                    TBWT = calculate_TBWT(R)

                    print(
                        f"=== Experiment Results: ACC = {ACC:.4f}, BWT = {BWT:.4f}, avg_CBWT = {avg_CBWT:.4f}, TBWT = {TBWT:.4f} ===")
                    results.append(
                        (loss_type, optimizer_type, depth, dropout_prob, ACC, BWT, avg_CBWT, TBWT))
    return results


def plot_results(results):
    # Aggregate results into a DataFrame for visualization
    df = pd.DataFrame(results, columns=[
                      "Loss", "Optimizer", "Depth", "Dropout", "ACC", "BWT", "avg_CBWT", "TBWT"])
    print("\nAggregated Experiment Results:")
    print(df)

    # Plot mean ACC by loss type
    df.groupby("Loss")["ACC"].mean().plot(
        kind='bar', title="Mean ACC by Loss Type")
    plt.ylabel("ACC")
    plt.show()

    # Plot mean BWT by loss type
    df.groupby("Loss")["BWT"].mean().plot(
        kind='bar', title="Mean BWT by Loss Type")
    plt.ylabel("BWT")
    plt.show()

    # Plot mean avg_CBWT by loss type
    df.groupby("Loss")["avg_CBWT"].mean().plot(
        kind='bar', title="Mean avg_CBWT by Loss Type")
    plt.ylabel("avg_CBWT")
    plt.show()

    # Plot mean TBWT by loss type
    df.groupby("Loss")["TBWT"].mean().plot(
        kind='bar', title="Mean TBWT by Loss Type")
    plt.ylabel("TBWT")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_results = run_experiments(device)
    plot_results(experiment_results)
