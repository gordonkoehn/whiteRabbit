# visuals/plot_policy_and_tradeoff.py

import matplotlib.pyplot as plt
from whitebit_model import WhiteBitNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Simulate cost per operation ===
PRECISION_COST = {
    'fp8': 1.0,
    'fp16': 2.0
}

# === Precision configs ===
POLICIES = {
    "FP8 Only": [True, True, True, True],
    "FP16 Only": [False, False, False, False],
    "Mixed (1st & last FP16)": [False, True, True, False]
}

# === Load dataset ===
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64)

def train_and_eval(policy, epochs=1):
    model = WhiteBitNet(precision_policy=policy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        break  # only 1 epoch, 1 batch for now

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            break  # 1 batch only
    accuracy = correct / total

    # Compute cost
    cost = sum(PRECISION_COST['fp8'] if p else PRECISION_COST['fp16'] for p in policy)
    return accuracy, cost

def plot_results(results):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Bar chart showing each policy
    for i, (name, policy) in enumerate(POLICIES.items()):
        ax[0].bar([f"L{j+1}" for j in range(len(policy))], [1 if p else 2 for p in policy],
                  alpha=0.6, label=name)
    ax[0].set_title("Precision Policy per Layer (1=FP8, 2=FP16)")
    ax[0].legend()

    # Plot 2: Accuracy vs Cost
    for name, (acc, cost) in results.items():
        ax[1].scatter(cost, acc, label=name, s=100)
    ax[1].set_xlabel("Compute Cost")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy vs Cost Trade-Off")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = {}
    for name, policy in POLICIES.items():
        print(f"Training with policy: {name}")
        acc, cost = train_and_eval(policy)
        results[name] = (acc, cost)

    plot_results(results)
