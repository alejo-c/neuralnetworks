import sys

import numpy as np
import matplotlib.pyplot as plt


def f(b0, b1, xi):
    return b0 + b1 * xi


def solve(values):
    x = np.array(values[:, 0]) # (array)
    y = np.array(values[:, 1]) # (array)
    xp = np.mean(x) # Estimated/Predicted value of x
    yp = np.mean(y) # Estimated/Predicted value of y
    xe = x - xp # Residuals error of x (array)
    ye = y - yp # Residuals error of y (array)
    sxeye = np.sum(np.multiply(xe, ye))
    sxe2 = np.sum(np.power(xe, 2)) # standarized error of x
    b1 = sxeye / sxe2 # Estimate of bi (slope)
    b0 = yp - b1 * xp # Y-Intercept point
    return x, y, b0, b1


def graph(x, y, b0, b1):
    xn = np.linspace(-100, 100)
    yn = f(b0, b1, xn)
    fig, ax =  plt.subplots()
    ax.plot(xn, yn)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Solve Simple Linear Regression Model')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True)
    ax.plot(xn, yn, color='red')
    ax.plot(x, y, 'o', color='blue')
    plt.show()


def main():
    values = np.loadtxt(sys.argv[1])
    graph(*solve(values))


if __name__ == '__main__':
    main()
