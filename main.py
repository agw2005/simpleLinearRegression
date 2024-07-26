import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('fertility_Diagnosis.csv')
x_train = data['age'].to_numpy()*18
y_train = data['numberOfHoursSpentSittingPerday'].to_numpy()


def rescale(arr):
    max_value = arr.max()
    return arr/max_value, max_value


def compute_derivatives(x, y, weight, bias):
    m = len(x)
    f_wb = weight * x + bias
    error = f_wb - y
    derivative_w = np.dot(error, x) / m
    derivative_b = error.mean()
    return derivative_w, derivative_b


def gradient_descent(x, y, alpha=0.001, weight=0, bias=0):
    weight = weight
    bias = bias
    learning_rate = alpha
    not_converged = True

    while not_converged:
        dw, db = compute_derivatives(x, y, weight, bias)

        temp_w = weight - learning_rate * dw
        temp_b = bias - learning_rate * db

        if (temp_w == weight) and (temp_b == bias):
            not_converged = False

        weight = temp_w
        bias = temp_b

    return weight, bias


def compute_regression_line(x, y, weight, bias, title=None, label_x=None, label_y=None):
    m = len(x)
    prediction_line = np.zeros(m)
    for i in range(m):
        prediction_line[i] = weight * x[i] + bias

    plt.plot(x, prediction_line, c='b', label='prediction')
    plt.scatter(x, y, marker='x', c='r', label='reality')
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.show()
    return None


x_train_rescaled, x_max = rescale(x_train)
y_train_rescaled, y_max = rescale(y_train)

w_rescaled, b_rescaled = gradient_descent(x_train_rescaled, y_train_rescaled)

w = w_rescaled * (y_max / x_max)
b = b_rescaled * y_max

compute_regression_line(x_train, y_train, w, b, title="Age vs. Hours Spent Sitting", label_x="Age", label_y="Hours Spent Sitting")
print(f"w = {w}\nb = {b}")

"""
100 examples, took 37 -- 41 seconds

w = -0.037766034512644465
b = 0.8615785876012092

Citation:
Gil,David and Girela,Jose. (2013). Fertility. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z01Z.
"""