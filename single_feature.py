import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def compute_cost(w, b, x: ndarray, y: ndarray):
    """
    Computes the cost function for linear regression model

    Args:
    w : weight
    b : bias
    x : feature examples
    y : labels

    Returns:
    total cost for the model to fit the data (x,y)
    """
    # Get the number of features
    m = len(x)
    cost = 0
    for i in range(m):
        error = w*x[i] + b - y[i]
        cost += error**2
    return cost/(2*m)

def cost_func_der(w, b, x : ndarray, y : ndarray):
    """
    Computes the partial derivates of the cost function (or the step size of the gradient descent)

    Args:
    w : weight
    b : bias
    x : feature examples
    y : labels

    Returns:
    dj_dw : the step size for w
    dj_db : the step size for b
    """

    # Get the number of features
    m = len(x)

    # Initialize the variables
    dj_dw = 0
    dj_db = 0

    # Compute the derivate
    for i in range(m):
        error = w*x[i] + b - y[i]
        dj_dw += error * x[i]
        dj_db += error

    return dj_dw/m, dj_db/m

def gradient_descent(x:ndarray, y:ndarray, w_in, b_in, alpha, iters, cost_func_der : cost_func_der):
    """
    Performs the gradient descent for linear regression

    Args:
    x               : feature examples
    y               : labels
    w_in            : initial weight
    b_in            : initial bias
    alpha           : learning rate
    iters           : number of iterations for gradient descent steps
    cost_func_der   : function to compute the step size or the gradient

    Returns:
    w : new value for weight
    b : new value for bias
    """
    w = w_in
    b = b_in
    for i in range(iters):
        dj_dw, dj_db = cost_func_der(w, b, x, y)
        w -= alpha*dj_dw
        b -= alpha*dj_db
    return w, b

def main():
    x = 2*np.random.rand(100,)
    y = 4*x + np.random.rand(100,) + 0.3*np.random.randint(6) 
    w, b = gradient_descent(x, y, 0, 0, 0.001, 100000, cost_func_der)

    # Compare these w and b to those returned by sklearn's Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(x.reshape(-1,1), y)
    print(f"w_from_sklearn: {lin_reg.coef_[0]} || b_from_sklearn: {lin_reg.intercept_}")
    print(f"check_w: {w} || check_b: {b}")

    # Make a prediction
    x_new = np.linspace(0, 2)
    y_hat = w*x_new + b

    fig = plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=15, c="b", marker="o")
    plt.plot(x_new, y_hat, "r-", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([0, 2, 0, 10])
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()