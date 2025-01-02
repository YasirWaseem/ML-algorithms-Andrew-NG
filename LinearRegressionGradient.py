import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def compute_cost(self, X, y):
        """Compute the cost function for linear regression."""
        m = len(y)
        predictions = X @ self.theta
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        """Fit the model to the training data using gradient descent."""
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for _ in range(self.iterations):
            # Gradient Descent
            gradient = (1 / m) * (X.T @ (X @ self.theta - y))
            self.theta -= self.learning_rate * gradient
            self.cost_history.append(self.compute_cost(X, y))

    def predict(self, X):
        """Predict the output for given input data."""
        return X @ self.theta

# Example Usage
if __name__ == "__main__":
    # Sample data (x: hours studied, y: exam scores)
    data = np.array([
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10]
    ])
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    # Add bias term to X
    X = np.c_[np.ones(X.shape[0]), X]

    # Create and train the model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)

    # Print results
    print("Theta (parameters):", model.theta)
    print("Cost after training:", model.cost_history[-1])

    # Plot the results
    plt.scatter(X[:, 1], y, color="blue", label="Data")
    plt.plot(X[:, 1], predictions, color="red", label="Linear Fit")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Scores")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()