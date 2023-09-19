from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mae_regression(guess: np.array, x: np.array, y: np.array, loss_values: list) -> float:
    """Median Absolute Error Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat = m * x + b
    # Get loss MAE
    mae = np.median(np.abs(y - y_hat))
    loss_values.append(mae)
    return mae



if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Initial guess of the parameters: [2, 2] (m, b).
    # It doesn’t have to be accurate but simply reasonable.
    initial_guess = np.array([3, 4])

    # Maximizing the probability for point to be from the distribution
    loss_values = []
    results = minimize(
        mae_regression,
        initial_guess,
        args=(x, y, loss_values),
        method="SLSQP",
        options={"disp": True})

    fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
    ax_loss.plot(range(len(loss_values)), loss_values)
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss Value')
    ax_loss.set_title('Loss Function Value over Iterations')
    plt.savefig('median_absolute_error_loss.png')

    
    print(results)
    print("Parameters: ", results.x)

    # Plot results
    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = results.x[0] * xx + results.x[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(xx, yy, "r--.", label="MAE")
    ax.legend(loc="best")

    plt.savefig("median_absolute_error_regression.png")
