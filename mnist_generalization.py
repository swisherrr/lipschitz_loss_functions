"""
Empirical test of Generalization on MNIST.
We test three loss functions (Cross-Entropy, MSE, MAE) to measure
how effectively they generalize to unseen data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Deterministic settings for exact replication
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
np.random.seed(42)

# Dataset S
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)

# 5000 training examples
DATA_SIZE = 5000
test_data = [mnist_train[i] for i in range(5000, 6000)] # 1000 test items

# Dataset S (Indices 0 to 4999)
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
def evaluate_generalization(loss_type="Cross-Entropy"):
    print(f"\nEvaluating Generalization using Loss Type: {loss_type}")
    print("-" * 60)
    
    # We must ensure all models start with the EXACT same neural weights
    torch.manual_seed(999)
    model = LeNetInspired()
    
    # Setting uniform Learning Rate to 0.5
    opt = optim.SGD(model.parameters(), lr=0.5)
    
    batch_size = 60
    epochs = 5
    
    metrics = []
    
    for epoch in range(epochs):
        model.train()
        
        # Consistent shuffling
        torch.manual_seed(epoch) 
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
            
        # End of Epoch: Record train and test metrics
        model.eval()
        with torch.no_grad():
            preds_train = model(S_X).argmax(dim=1)
            acc_train = (preds_train == S_y).float().mean().item()
            
            preds_test = model(test_X).argmax(dim=1)
            acc_test = (preds_test == test_y).float().mean().item()
            
        metrics.append((acc_train, acc_test))
        
        print(f"Epoch {epoch+1:2d} | Train Acc: {acc_train*100:5.2f}% | Test Acc: {acc_test*100:5.2f}%")
        
    return metrics

ce_metrics = evaluate_generalization("Cross-Entropy")
mse_metrics = evaluate_generalization("Squared Loss (MSE)")
mae_metrics = evaluate_generalization("Mean Absolute Error (MAE)")

print("\n" + "="*85)
print("SUMMARY: OUT-OF-SAMPLE GENERALIZATION (TEST ACCURACY)")
print("="*85)
print(f"{'Epoch':<6} | {'Cross-Entropy Acc':<22} | {'MSE Acc':<22} | {'MAE Acc':<22}")
print("-" * 85)
for epoch in range(5):
    ce_acc = ce_metrics[epoch][1] * 100
    mse_acc = mse_metrics[epoch][1] * 100
    mae_acc = mae_metrics[epoch][1] * 100
    print(f"{epoch+1:<6} | {ce_acc:<20.2f}% | {mse_acc:<20.2f}% | {mae_acc:<20.2f}%")
