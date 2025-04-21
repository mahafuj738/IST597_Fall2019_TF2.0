import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

###########################
# Custom Normalization Modules
###########################


class BatchNormManual(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        if x.dim() == 2:
            m = x.mean(dim=0, keepdim=True)
            v = x.var(dim=0, unbiased=False, keepdim=True)
            xh = (x - m) / torch.sqrt(v + self.eps)
            return self.gamma * xh + self.beta
        elif x.dim() == 4:
            m = x.mean(dim=[0, 2, 3], keepdim=True)
            v = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
            xh = (x - m) / torch.sqrt(v + self.eps)
            return self.gamma.view(1, -1, 1, 1)*xh + self.beta.view(1, -1, 1, 1)
        else:
            raise NotImplementedError


class LinearWeightNorm(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.v = nn.Parameter(torch.randn(out_f, in_f))
        self.g = nn.Parameter(torch.ones(out_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        vn = self.v.norm(dim=1, keepdim=True)
        w = (self.g.unsqueeze(1)/vn) * self.v
        return F.linear(x, w, self.bias)


class LayerNormManual(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)
        v = x.var(dim=1, unbiased=False, keepdim=True)
        xh = (x - m) / torch.sqrt(v + self.eps)
        return self.gamma * xh + self.beta

###########################
# Model Definitions
###########################


class NetBaseline(nn.Module):
    """Simple MLP with no normalization"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class NetBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = BatchNormManual(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


class NetLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.ln1 = LayerNormManual(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)


class NetWN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = LinearWeightNorm(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

###########################
# Training & Evaluation
###########################


def train_one_epoch(model, device, loader, optimizer):
    model.train()
    loss_sum, gn_sum = 0.0, 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        if isinstance(model, NetWN):
            gn = model.fc1.v.grad.norm().item()
        else:
            gn = model.fc1.weight.grad.norm().item() if hasattr(
                model.fc1, 'weight') else model.fc1.weight.grad.norm().item()
        optimizer.step()
        loss_sum += loss.item()
        gn_sum += gn
    return loss_sum/len(loader), gn_sum/len(loader)


def test_model(model, device, loader):
    model.eval()
    sum_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            sum_loss += F.cross_entropy(out, target, reduction='sum').item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    sum_loss /= len(loader.dataset)
    return sum_loss, correct/len(loader.dataset)

###########################
# Main Experiment
###########################


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True,
                              download=True, transform=transform),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    # Instantiate models & optimizers
    net_base = NetBaseline().to(device)
    opt_base = optim.Adam(net_base.parameters(), lr=1e-3)

    net_bn = NetBN().to(device)
    opt_bn = optim.Adam(net_bn.parameters(), lr=1e-3)

    net_ln = NetLN().to(device)
    opt_ln = optim.Adam(net_ln.parameters(), lr=1e-3)

    net_wn = NetWN().to(device)
    opt_wn = optim.Adam(net_wn.parameters(), lr=1e-3)

    models = {
        'NoNorm':    (net_base, opt_base),
        'BatchNorm': (net_bn,   opt_bn),
        'LayerNorm': (net_ln,   opt_ln),
        'WeightNorm': (net_wn,   opt_wn),
    }

    # Metrics
    metrics = {name: {'train_loss': [], 'test_acc': [], 'grad_norm': []}
               for name in models}
    epochs = 50

    for ep in range(1, epochs+1):
        print(f"\n===== Epoch {ep} =====")
        for name, (model, opt) in models.items():
            print(f"\n--- {name} ---")
            tl, gn = train_one_epoch(model, device, train_loader, opt)
            print(f"Train loss: {tl:.4f}, Avg grad‑norm: {gn:.4f}")
            test_l, test_a = test_model(model, device, test_loader)
            print(f"Test  loss: {test_l:.4f}, Acc: {100*test_a:.2f}%")
            metrics[name]['train_loss'].append(tl)
            metrics[name]['test_acc'].append(test_a)
            metrics[name]['grad_norm'].append(gn)

    # Optional: Compare custom BatchNorm with PyTorch's built-in BatchNorm1d on a sample tensor.
    x_sample = torch.randn(10, 256).to(device)
    bn_custom = BatchNormManual(256).to(device)
    bn_builtin = nn.BatchNorm1d(256, eps=1e-5, affine=True).to(device)
    # Initialize built-in parameters to match the custom BN for a fair comparison.
    with torch.no_grad():
        bn_builtin.weight.copy_(bn_custom.gamma)
        bn_builtin.bias.copy_(bn_custom.beta)
    out_custom = bn_custom(x_sample)
    out_builtin = bn_builtin(x_sample)
    diff = torch.abs(out_custom - out_builtin).mean().item()
    print(
        f"\nMean difference between custom BN and built-in BN (should be very small): {diff:.6f}")

    # ----------------------------------------------------------------
    # Compare custom LayerNorm with built‑in LayerNorm
    x_sample = torch.randn(10, 256).to(device)
    ln_custom = LayerNormManual(256, eps=1e-5).to(device)
    ln_builtin = nn.LayerNorm(
        256, eps=1e-5, elementwise_affine=True).to(device)
    # copy gamma/beta
    with torch.no_grad():
        ln_builtin.weight.copy_(ln_custom.gamma)
        ln_builtin.bias.copy_(ln_custom.beta)

    out_ln_custom = ln_custom(x_sample)
    out_ln_builtin = ln_builtin(x_sample)
    diff_ln = (out_ln_custom - out_ln_builtin).abs().mean().item()
    print(f"Mean difference between custom LN and built‑in LN: {diff_ln:.6f}")

    # ----------------------------------------------------------------
    # Compare custom WeightNorm with built‑in weight_norm on a Linear layer
    # e.g. same input dim as your NetWN
    x_linear = torch.randn(10, 28*28).to(device)
    wn_custom = LinearWeightNorm(28*28, 256).to(device)
    # create a regular linear and wrap it
    linear_built = nn.Linear(28*28, 256, bias=True).to(device)
    linear_built = torch.nn.utils.weight_norm(linear_built, name='weight')

    # copy v and g into built‑in weight_norm
    with torch.no_grad():
        # compute what custom wn layer’s weight would be
        v = wn_custom.v            # shape [256, 784]
        g = wn_custom.g.unsqueeze(1)            # shape [256]
        v_norm = v.norm(dim=1, keepdim=True)        # [256,1]
        w_cust = (g.unsqueeze(1) / v_norm) * v      # [256,784]
        # now copy into built‑in
        linear_built.weight_v.copy_(v)
        linear_built.weight_g.copy_(g)
        linear_built.bias.copy_(wn_custom.bias)

    out_wn_custom = wn_custom(x_linear)
    out_wn_builtin = linear_built(x_linear)
    diff_wn = (out_wn_custom - out_wn_builtin).abs().mean().item()
    print(f"Mean difference between custom WN and built‑in WN: {diff_wn:.6f}")

    # Plotting
    epochs_range = list(range(1, epochs+1))

    # Training Loss
    plt.figure()
    for name in metrics:
        plt.plot(epochs_range, metrics[name]['train_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test Accuracy
    plt.figure()
    for name in metrics:
        plt.plot(epochs_range, [
                 a*100 for a in metrics[name]['test_acc']], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
