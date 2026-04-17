"""
Averaged Empirical test of Generalization on MNIST.
Runs the model training multiple times with mathematically varied random seeds
to calculate the average Peak Accuracy and Average Final Accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


torch.backends.cudnn.deterministic = True

# Dataset S
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)

DATA_SIZE = 5000
test_data = [mnist_train[i] for i in range(5000, 6000)]
S = [mnist_train[i] for i in range(DATA_SIZE)]

S_X = torch.stack([item[0] for item in S])
S_y = torch.tensor([item[1] for item in S])

test_X = torch.stack([item[0] for item in test_data])
test_y = torch.tensor([item[1] for item in test_data])


# Architecture
class LeNetInspired(nn.Module):
    def __init__(self):
        super(LeNetInspired, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500) 
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training Algorithm
def run_single_experiment(loss_type, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = LeNetInspired()
    opt = optim.SGD(model.parameters(), lr=0.5)
    
    batch_size = 60
    epochs = 5
    
    peak_acc = 0.0
    final_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(DATA_SIZE)
        
        for i in range(0, DATA_SIZE, batch_size):
            batch_idx = indices[i : i + batch_size]
            opt.zero_grad()
            out = model(S_X[batch_idx])
            
            if loss_type == "Cross-Entropy":
                loss = F.cross_entropy(out, S_y[batch_idx])
            elif loss_type == "Squared Loss (MSE)":
                y_onehot = F.one_hot(S_y[batch_idx], num_classes=10).float()
                loss = F.mse_loss(torch.softmax(out, dim=1), y_onehot)
            else: # Mean Absolute Error (MAE)
                y_onehot = F.one_hot(S_y[batch_idx], num_classes=10).float()
                loss = F.l1_loss(torch.softmax(out, dim=1), y_onehot)
                
            loss.backward()
            opt.step()
            
        model.eval()
        with torch.no_grad():
            preds_test = model(test_X).argmax(dim=1)
            acc_test = (preds_test == test_y).float().mean().item()
            
        if acc_test > peak_acc:
            peak_acc = acc_test
        if epoch == epochs - 1:
            final_acc = acc_test
            
    return peak_acc * 100, final_acc * 100

def get_averages(loss_type, num_runs=5):
    print(f"\nEvaluating Average Generalization: {loss_type}")
    print("-" * 60)
    peaks = []
    finals = []
    
    for run in range(num_runs):
        # We stagger the seed heavily to ensure totally different initial trajectory conditions
        p, f = run_single_experiment(loss_type, seed=(run * 1337 + 42))
        peaks.append(p)
        finals.append(f)
        print(f"  Run {run+1}/{num_runs} -> Peak Acc: {p:<6.2f}% | Final Acc: {f:<6.2f}%")
        
    avg_peak = np.mean(peaks)
    avg_final = np.mean(finals)
    
    return avg_peak, avg_final



NUM_RUNS = 15 # num of trials to average over

ce_peak, ce_final = get_averages("Cross-Entropy", NUM_RUNS)
mse_peak, mse_final = get_averages("Squared Loss (MSE)", NUM_RUNS)
mae_peak, mae_final = get_averages("Mean Absolute Error (MAE)", NUM_RUNS)

print("\n" + "="*85)
print(f"AVERAGED OUT-OF-SAMPLE GENERALIZATION ({NUM_RUNS} RANDOMIZED RUNS)")
print("="*85)
print(f"{'Loss Function':<30} | {'Avg Peak Accuracy':<22} | {'Avg Final Accuracy':<22}")
print("-" * 85)
print(f"{'Cross-Entropy (Non-Lipschitz)':<30} | {ce_peak:<20.2f}% | {ce_final:<20.2f}%")
print(f"{'Squared Loss MSE (Lipschitz)':<30} | {mse_peak:<20.2f}% | {mse_final:<20.2f}%")
print(f"{'Mean Abs Err MAE (Lipschitz)':<30} | {mae_peak:<20.2f}% | {mae_final:<20.2f}%")
