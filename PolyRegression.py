# Poly Regression with PyTorch
# For example, to predict x**3, 
# From the perspective of a neural network, it can be regarded as three input features. 
# Different orders of x are regarded as a separate input feature.
# So there are three inputs, and only one output

import torch
import numpy as np
import random

class PolyRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(PolyRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)

def generate_data(batch_size=32):
    # Generate normally distributed random numbers, and add a dimension
    # 32 rows, 1 column
    x = torch.randn(batch_size).unsqueeze(1)
    # f(x) = 2 * x^3 + 3 * x^2 + 4 * x + 5
    # 32 rows, 1 column
    y = 2.0 * x**3 + 3.0 * x**2 + 4.0 * x + 5.0 + random.random()/10

    # Take x, x**2, x**3 as inputs
    # 32 rows, 3 columns
    x_data = torch.cat([x**i for i in range(1, 4)], 1)
    return (x_data, y)

def train(epoch=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PolyRegression(3).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(epoch):
        x, y = generate_data(32)
        x = x.to(device)
        y = y.to(device)

        predicted = model(x)
        loss = loss_func(predicted, y)
        optimizer.zero_grad()
        # BP
        loss.backward()
        # Optimize parameters
        optimizer.step()

        print(f'Epoch {i}, loss {loss.item()}')

    # Final Results
    print(f'Weight: {model.linear.weight.data}')
    print(f'Bias: {model.linear.bias.data}')


if __name__ == '__main__':
    train(5000)

