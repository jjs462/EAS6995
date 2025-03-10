import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

'''
Functions to look at that may be useful:np.sum()
np.where()
np.maximum()
np.log()
np.exp()
np.argmax()
np.dot()
.append()
np.random.choice()

For torch tensors:
X.view()
X.numpy()
X.item()
dataset.targets
dataset.data
'''

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
def fully_connected_backward(X, Z, W, dZ):
    """
    NOTE CLARIFICATION HERE; dZ is an input instead of Y
    Compute gradients for the fully connected (linear) layer.
    X: Input data (Nxd)
    Z: Output of the layer before activation (logits, NxK)
    W: Weight matrix (dxK)
    dZ: Gradient of the loss with respect to Z (from the next layer)
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
def softmax_backward(S, Y):
    """
    NOTE THE CORRECTION/EFFICIENCY GAIN HERE in using softmax output instead of Z
    Compute the gradient of the loss with respect to softmax output.
    S: Output of softmax 
    Y: True labels (one-hot encoded)
    """
    dZ = None  # TODO: Compute dZ for softmax (S - Y)
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
    # NOTE THE CORRECTION HERE! I HAD it done using torch but needs to be numpy
    # Note also that this is not using the specific methods I had mentioned for
    #   weight initialization (e.g. Xavier or He), this is just random
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
    b1 = np.zeros(hidden_dim1)
    W2 = #TODO
    b2 = #TODO
    W3 = #TODO
    b3 = #TODO
    
    # ADD THESE to save training and test loss, accuracy
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []
    
    # Loop through epochs
    for epoch in range(epochs):
        epoch_loss = 0
        test_epoch_loss = 0
        correct_predictions = 0
        total_correct_predictions = 0
        total_samples = 0

        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            # Flatten images to vectors
            X_batch = #TODO  # Flatten images to vector
            Y_batch = torch.eye(output_dim)[Y_batch]  # Map label indices to corresponding one-hot encoded vectors
            
            # CONVERT TORCH TENSORS to numpy
            X = # TODO
            y = # TODO

            # --- TODO: Implement the forward pass ---
            Z1 = #TODO
            A1 = #TODO
            Z2 = #TODO
            A2 = #TODO
            Z3 = #TODO
            Y_pred = #TODO
            
            # --- TODO: Implement loss computation ---
            loss = ___

            epoch_loss = #TODO

            # --- TODO: Implement backward pass ---
            dZ3 = #TODO
            dW3, db3, dA2 = #TODO
            dZ2 = #TODO
            dW2, db2, dA1 = #TODO
            dZ1 = #TODO
            dW1, db1, dX = #TODO

            # --- TODO: Implement weight update ---
            W1, b1 = update_weights(W1, b1, dW1, db1, learning_rate)
            W2, b2 = update_weights(W2, b2, dW2, db2, learning_rate)
            W3, b3 = update_weights(W3, b3, dW3, db3, learning_rate)

            # Track accuracy
            correct_predictions = #for this batch; TODO
            total_correct_predictions = #for the entire epoch; TODO
            total_samples = #for entire epoch; TODO

        # Print out the progress - CLARIFIED
        train_accuracy = total_correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {train_accuracy * 100}%")
        
        # Save the training loss and accuracy for each epoch to plot later

        # TODO: For every 100 epochs, get the validation loss and error
        # FREQUENCY OF THIS IS CHANGED FROM EVERY 1000 to EVERY 100
        
        # Save the test loss and accuracy for every 100th epoch to plot later
        
    return training_loss, training_accuracy, test_loss, test_accuracy
    print("Training complete!")

# Main function
def main():
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # Start training
    training_loss, training_accuracy, test_loss, test_accuracy = train(train_loader, test_loader, epochs=1000, learning_rate=0.1)
    
    
    # PLOT TRAINING LOSS AND TEST LOSS ON ONE SUBPLOT (epoch vs loss)
    # PLOT TRAINING ACCURACY AND TEST ACCURACY ON A SECOND SUBPLOT (epoch vs accuracy)
    
    epochs_train = list(range(1, len(training_loss) + 1))  # Epochs for training loss (1, 2, ..., N)
    epochs_test = list(range(100, (len(test_loss) + 1) * 100, 100))  # Epochs for test loss (100, 200, ..., N*100)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Training and Test Loss on the first subplot
    ax1.plot(..., ..., label='Training Loss', color='blue', marker='o')
    ax1.plot(..., ..., label='Test Loss', color='red', marker='x')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Training and Test Accuracy on the second subplot
    ax2.plot(..., ..., label='Training Accuracy', color='blue', marker='o')
    ax2.plot(..., ..., label='Test Accuracy', color='red', marker='x')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
