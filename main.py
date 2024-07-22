import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.5, 1.0, 2.8, 0.4, 1.3, 2.0, 3.1, 4.2, 0.9, 2.5, 3.6, 4.0, 2.2, 3.3, 1.8])
y_train = np.array([3.6, 2.8, 5.4, 1.9, 2.9, 4.3, 6.2, 8.0, 2.5, 5.0, 7.3, 8.4, 4.7, 6.5, 4.0])

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Biaya iklan dan laba bersih")
plt.xlabel("Biaya iklan perusahaan")
plt.ylabel("Laba bersih perusahaan")
plt.show()


def compute_derivatives(x, y, weight, bias):
    m = len(x)
    derivative_w = 0
    derivative_b = 0
    for i in range(m):
        f_wb = weight * x[i] + bias
        derivative_of_cost_wrt_w = (f_wb - y[i]) * x[i]
        derivative_of_cost_wrt_b = (f_wb - y[i])
        derivative_w += derivative_of_cost_wrt_w
        derivative_b += derivative_of_cost_wrt_b
    derivative_w /= m
    derivative_b /= m
    return derivative_w, derivative_b


def gradient_descent(x, y, weight=0, bias=0, alpha=0.1):
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


def compute_line(x, weight, bias):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = weight * x[i] + bias
    return f_wb


w, b = gradient_descent(x_train, y_train)
regression_line = compute_line(x_train, w, b)
plt.plot(x_train, regression_line, c='b', label='prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='reality')
plt.title("Biaya iklan dan laba bersih")
plt.xlabel("Biaya iklan perusahaan")
plt.ylabel("Laba bersih perusahaan")
plt.legend()
plt.show()
