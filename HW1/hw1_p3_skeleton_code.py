import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Load the FashionMNIST dataset 
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])  # Only convert to tensor

    # Download the dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # --- TODO: Normalize data manually to the range [0, 1] ---
    # Normalize to [0, 1]
    # train_dataset.data = ___
    # test_dataset.data = ___

    # Subsampling: 50% from each class
    train_indices = subsample_50_percent_per_class(train_dataset)
    train_subset = Subset(train_dataset, train_indices)

    # DataLoader for batching
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to perform subsampling 50% from each class
def subsample_50_percent_per_class(dataset):
    """
    Subsample 50% of the data from each class.
    dataset: The full dataset (e.g., FashionMNIST)
    Returns: A list of indices for the subsampled dataset
    """
    # --- TODO: Implement subsampling logic here ---
    sampled_indices = []

    return sampled_indices


# Forward pass for Fully Connected Layer
def fully_connected_forward(X, W, b):
    """
    Perform forward pass for a fully connected (linear) layer.
    X: Input data
    W: Weight matrix
    b: Bias vector
    """
    Z = None  # TODO: Compute the linear transformation (X * W + b)
    return Z

# Forward pass for ReLU activation
def relu_forward(Z):
    """
    ReLU activation function forward pass.
    Z: Linear output (input to ReLU)
    """
    A = None  # TODO: Apply ReLU function (element-wise)
    return A

# Forward pass for Softmax activation
def softmax_forward(Z):
    """
    Softmax activation function forward pass.
    Z: Output logits (before softmax)
    """
    exp_z = None  # TODO: Apply softmax function (numerical stability)
    output = None  # TODO: Normalize exp_z to get the softmax output
    return output

# Backward pass for Fully Connected Layer (Linear)
def fully_connected_backward(X, Z, W, Y):
    """
    Compute gradients for the fully connected (linear) layer.
    X: Input data
    Z: Output of the layer before activation (logits)
    W: Weight matrix
    Y: True labels (for loss gradient calculation)
    """
    dW = None  # TODO: Compute gradient of weights (X^T * dZ)
    db = None  # TODO: Compute gradient of bias (sum of dZ)
    dZ = None  # TODO: Compute gradient of loss with respect to Z (for backpropagation)
    return dW, db, dZ

# Backward pass for ReLU activation
def relu_backward(Z, dA):
    """
    Compute the gradient for ReLU activation.
    Z: Input to ReLU (before activation)
    dA: Gradient of the loss with respect to activations (from the next layer)
    """
    dZ = None  # TODO: Compute dZ for ReLU (gradient is 0 for Z <= 0 and dA for Z > 0)
    return dZ

# Backward pass for Softmax Layer
def softmax_backward(Z, Y):
    """
    Compute the gradient of the loss with respect to softmax output.
    Z: Output logits (before softmax)
    Y: True labels (one-hot encoded)
    """
    dZ = None  # TODO: Compute dZ for softmax (Z - Y)
    return dZ

# Weight update function (gradient descent)
def update_weights(weights, biases, grads_W, grads_b, learning_rate=0.01):
    """
    --- TODO: Implement the weight update step ---
    weights: Current weights
    biases: Current biases
    grads_W: Gradient of the weights
    grads_b: Gradient of the biases
    learning_rate: Learning rate for gradient descent
    """
    pass


# Define the neural network 
def train(train_loader, test_loader, epochs=10000, learning_rate=0.01):
    # Initialize weights and biases
    input_dim = #TODO
    hidden_dim1 = 128   #could set differently
    hidden_dim2 = 64    #could set differently
    output_dim = # TODO
    
    # Initialize weights randomly
    W1 = torch.randn(input_dim, hidden_dim1, requires_grad=False) * 0.01
    b1 = torch.zeros(hidden_dim1, requires_grad=False)
    W2 = #TODO
    b2 = #TODO
    W3 = #TODO
    b3 = #TODO
    
    # Loop through epochs
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            # Flatten images to vectors
            X_batch = #TODO  # Flatten images to vector
            Y_batch = torch.eye(output_dim)[Y_batch]  # Map label indices to corresponding one-hot encoded vectors

            # --- TODO: Implement the forward pass ---
            Z1 = #TODO
            A1 = #TODO
            Z2 = #TODO
            A2 = #TODO
            Z3 = #TODO
            Y_pred = #TODO
            
            # --- TODO: Implement loss computation ---
            # loss = ___

            epoch_loss = #TODO

            # --- TODO: Implement backward pass ---
            dZ3 = #TODO
            dW3, db3, dA2 = #TODO
            dZ2 = #TODO
            dW2, db2, dA1 = #TODO
            dZ1 = #TODO
            dW1, db1, _ = #TODO

            # --- TODO: Implement weight update ---
            W1, b1 = update_weights(W1, b1, dW1, db1, learning_rate)
            W2, b2 = update_weights(W2, b2, dW2, db2, learning_rate)
            W3, b3 = update_weights(W3, b3, dW3, db3, learning_rate)

            # Track accuracy
            correct_predictions = #TODO
            total_samples = #TODO

        # Print out the progress
        train_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {train_accuracy * 100}%")

        # TODO: For every 1000 epochs, get the validation loss and error
        
    print("Training complete!")

# Main function
def main():
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # Start training
    train(train_loader, test_loader, epochs=10000, learning_rate=0.1)

if __name__ == "__main__":
    main()
