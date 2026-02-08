import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.r_squared = None
        self.mse = None

    def fit(self, X, y):
        X_copy = X.copy()
        X_copy.insert(0, 'intercept', 1)
        X_mat = X_copy.values
        y_mat = y.values.reshape(-1, 1)
        
        # Normal Equation: theta = (X^T * X)^-1 * X^T * y
        xtx_inv = np.linalg.inv(np.dot(X_mat.T, X_mat))
        beta = np.dot(xtx_inv, np.dot(X_mat.T, y_mat))
        
        self.intercept = beta[0][0]
        self.coefficients = beta[1:].flatten()
        
    def predict(self, X):
        return np.dot(X.values, self.coefficients) + self.intercept

    def evaluate(self, X_test, y_test):
        """
        Tests the model on unseen data and returns R2 and Mean Squared Error.
        """
        y_pred = self.predict(X_test)
        y_true = y_test.values
        
        # Calculate R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((y_true - y_pred) ** 2)
        
        print(f"--- Test Results ---")
        print(f"R-squared: {r2:.4f}")
        print(f"MSE:       {mse:.2f}")
        

        self.r_squared = r2
        self.mse = mse

        return r2, mse

    def plot_results(self, X, y, title="Actual vs Predicted"):
        y_pred = self.predict(X)
        plt.figure(figsize=(8, 5))
        
        if X.shape[1] == 1:
            plt.scatter(X.iloc[:, 0], y, color='blue', label='Actual')
            plt.plot(X.iloc[:, 0], y_pred, color='red', label='Prediction')
        else:
            plt.scatter(y, y_pred, alpha=0.6, color='purple')
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual Y')
            plt.ylabel('Predicted Y')
            
        plt.title(title)
        plt.legend()
        plt.show()