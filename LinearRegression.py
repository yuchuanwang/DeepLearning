# Linear Regression with PyTorch

import torch 
import numpy as np
import random

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # Both input and output are 1-dimensional
        self.linear = torch.nn.modules.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)
    
def generate_data_1(batch_size=32):
    # Generate normally distributed input
    x = torch.randn(batch_size)
    # The target function is: y = 3x + 4
    y = 3.0 * x + 4.0 + random.randint(-1, 1)

    return (x, y)

def train_1(epoch=500):
    # Using CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LinearRegression().to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epoch):
        x, y = generate_data_1()
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # Move tensor to device
        x = x.to(device)
        y = y.to(device)
        predicted = model(x).to(device)

        # PyTorch standard process
        loss = loss_func(predicted, y).to(device)
        optimizer.zero_grad()
        # BP
        loss.backward()
        # Optimize parameters
        optimizer.step()

        print(f'Epoch {i}, loss {loss.item()}')

    # The final results
    print(f'Weight: {model.linear.weight.data}')
    print(f'Bias: {model.linear.bias.data}')

if __name__ == '__main__':
    train_1(5000)
