import torch
import numpy as np
import learner as ln
import pandas as pd
from learner.utils import grad

def main():
    branch_size = [441,200,100,50,25,1]
    trunk_size = [3,6,9,6,3,1]
    activation = 'relu'
    net = ln.nn.DeepONet(branch_size = branch_size, trunk_size = trunk_size, activation = activation)
    A = torch.randn(3,441)
    B = torch.randn(3,3)
    X = [[],[]]
    X[0] = A    
    X[1] = B
    Z = X[1].requires_grad_(True)
    C = net(X)
    print(X[0].shape)
    print(X[1].shape)
    print(C.shape)
    U = grad(C,Z)
    print(U)
if __name__ == '__main__':
    main()
