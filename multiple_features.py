import numpy as np
from numpy import ndarray
from sklearn.linear_model import LinearRegression

def compute_cost(w : ndarray, b, X : ndarray, y : ndarray):
    # no. of instances
    m = len(X)
    cost = 0
    for i in range(m):
        error = w@X[i] + b - y[i]
        cost += error**2
    return cost/(2*m)

def cost_func_der(w:ndarray, b, X:ndarray, y:ndarray):
    m, n = X.shape # no of instances, no of features

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        error = w@X[i] + b - y[i]
        for j in range(n):
            dj_dw[j] += error*X[i, j]
        dj_db += error
    
    return dj_dw/m, dj_db/m

def gradient_descent(X:ndarray, y:ndarray, w_in:ndarray, b_in, alpha, iters, cost_func_der:cost_func_der):
    w = w_in
    b = b_in
    for i in range(iters):
        dj_dw, dj_db = cost_func_der(w, b, X, y)
        w -= alpha*dj_dw
        b -= alpha*dj_db
    return w, b

def main():
    X = 2 * np.random.rand(100, 3)
    y = np.random.rand(100,) + 0.3*np.random.randint(6)
    w, b = gradient_descent(X, y, np.zeros((3,)), 0, 0.001, 100000, cost_func_der)

    # Compare these w and b to those returned by sklearn's Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(f"w_from_sklearn: {lin_reg.coef_} || b_from_sklearn: {lin_reg.intercept_}")
    print(f"check_w: {w} || check_b: {b}\n")

    # Make the predictions
    X_new = X[:5]
    y_new = y[:5]
    y_pred = X_new@w + b
    y_pred_sk = lin_reg.predict(X_new)
    
    for i in range(len(y_new)):
        print(f"Target: {y_new[i]:.3f}, Prediction: {y_pred[i]:.3f}, lin_reg Prediction: {y_pred_sk[i]:.3f}")

if __name__ == "__main__":
    main()

