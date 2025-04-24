from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUCellCustom(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combine input+hidden in one linear for each gate
        self.lin_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # x: (batch, input_size), h_prev: (batch, hidden_size)
        cat = torch.cat([h_prev, x], dim=1)
        z = torch.sigmoid(self.lin_z(cat))
        r = torch.sigmoid(self.lin_r(cat))
        cat2 = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.lin_h(cat2))
        h = (1 - z) * h_prev + z * h_tilde
        return h


class MGUCellCustom(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        cat = torch.cat([h_prev, x], dim=1)
        f = torch.sigmoid(self.lin_f(cat))
        cat2 = torch.cat([f * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.lin_h(cat2))
        h = (1 - f) * h_prev + f * h_tilde
        return h


class StackedRNN(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        Cell = GRUCellCustom if cell_type == 'GRU' else MGUCellCustom
        self.layers = nn.ModuleList([
            Cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        # init hidden states
        h = [x.new_zeros(batch_size, layer.hidden_size)
             for layer in self.layers]

        for t in range(seq_len):
            inp = x[:, t, :]
            for i, layer in enumerate(self.layers):
                h[i] = layer(inp, h[i])
                inp = h[i]
        # final output
        out = self.classifier(h[-1])  # (batch, num_classes)
        return out


class SafeImageFolder(ImageFolder):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = self.loader(path)
        except Exception:
            # corrupted → skip to the next (wrap around if needed)
            return self.__getitem__((idx + 1) % len(self.samples))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# # build ImageFolder but only keep truly valid images
# full = datasets.ImageFolder(
#     'data/notMNIST',
#     transform=transform,
#     is_valid_file=is_valid_file
# )
full = SafeImageFolder('data/notMNIST', transform=transform)
# full = datasets.ImageFolder('data/notMNIST', transform=transform)
# split 90% train / 10% test
n_train = int(0.9 * len(full))
train_ds, test_ds = random_split(full, [n_train, len(full) - n_train])
train_loader = DataLoader(train_ds, batch_size=128,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds,  batch_size=128,
                         shuffle=False, num_workers=4)


def run_trial(cell_type, hidden_size, num_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedRNN(cell_type, input_size=28*28, hidden_size=hidden_size,
                       num_layers=num_layers, num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = [], []
    for epoch in range(1, 21):
        # Training
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            # flatten each frame as sequence length=1
            xb = xb.view(xb.size(0), -1, )
            xb = xb.unsqueeze(1)            # (batch, seq=1, 784)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc.append(correct/total)
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.view(xb.size(0), -1, ).unsqueeze(1).to(device)
                yb = yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        test_acc.append(correct/total)

    return train_acc, test_acc


def compare_models():
    settings = [
        (cell, h, nl)
        for cell in ('GRU', 'MGU')
        for h in (50, 128, 256, 512)
        for nl in (1, 2, 3, 4)
    ]
    results = {}
    for cell_type, h, nl in settings:
        all_train, all_test = [], []
        for trial in range(3):
            t_acc, v_acc = run_trial(cell_type, h, nl)
            all_train.append(t_acc)
            all_test.append(v_acc)
        results[cell_type] = (all_train, all_test)

    return results


# Run all experiments
results = compare_models()

# Summarize final accuracies
print(f"{'Config':<20s}  {'Mean Acc (%)':>12s}  {'Std Acc (%)':>12s}")
for config, (_trials_t, trials_v) in results.items():
    finals = [v[-1] for v in trials_v]      # pick the 20th epoch
    mean_acc = np.mean(finals) * 100
    std_acc = np.std(finals) * 100
    print(f"{config:<20s}  {mean_acc:12.2f}  {std_acc:12.2f}")


# Plot
for cell_type, (trials_t, trials_v) in results.items():
    plt.figure()
    for t_acc, v_acc in zip(trials_t, trials_v):
        plt.plot(t_acc, '--',  alpha=0.5)
        plt.plot(v_acc, '-',   alpha=0.5)
    plt.title(f'{cell_type}: train (dashed) vs test (solid)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['trial1', 'trial1_val', 'trial2',
               'trial2_val', 'trial3', 'trial3_val'])
    plt.show()
