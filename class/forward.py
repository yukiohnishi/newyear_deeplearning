import numpy as np
import matplotlib.pyplot as plt
import argparse

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def forward(X, W1, W2):
    Z = sigmoid(X.dot(W1))
    A = Z.dot(W2)
    return sigmoid(A)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("x1")
    p.add_argument("x2")
    args = p.parse_args()
    print("input x1 is %d" % int(args.x1))
    print("input x2 is %d" % int(args.x2))
    X = np.array([int(args.x1), int(args.x2)])
    W1 = np.array([[1, 1],[1, 1]])
    W2 = np.array([1, 1])
    print("probability is %.3f" % forward(X, W1, W2))
