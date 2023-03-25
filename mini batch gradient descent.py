import numpy as np
import matplotlib.pyplot as plt
import timeit

def mini_batch_gradient_descent(X, y, alpha=0.01, epochs=100, batch_size=32):
    """
    Implementation of mini-batch gradient descent.

    Parameters:
    X: numpy array of shape (m, n)
       Training examples, where m is the number of examples and n is the number of features.
    y: numpy array of shape (m, 1)
       Target values.
    alpha: float
       Learning rate.
    epochs: int
       Number of passes over the training set.
    batch_size: int
       Size of mini-batch.

    Returns:
    theta: numpy array of shape (n, 1)
       Optimal parameters.
    loss: list
       Training loss at each epoch.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    loss = []

    for epoch in range(epochs):
        # Shuffle the training set
        permutation = np.random.permutation(m)
        X = X[permutation]
        y = y[permutation]

        for i in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Compute gradient
            grad = 1/batch_size * X_batch.T @ (X_batch @ theta - y_batch)

            # Update parameters
            theta = theta - alpha * grad

        # Compute training loss
        loss.append(1/m * np.sum((X @ theta - y)**2))

    return theta, loss

# Generate synthetic data
m = 1000
n = 10
X = np.random.randn(m, n)
y = X @ np.random.randn(n, 1)
# Train model using mini-batch gradient descent
start_time = timeit.default_timer()
theta, loss = mini_batch_gradient_descent(X, y)
end_time = timeit.default_timer()
print(f"Training time: {end_time - start_time:.4f} seconds")

# Plot training loss over time
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()


