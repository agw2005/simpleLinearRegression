import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.5, 1.0, 2.8, 0.4, 1.3, 2.0, 3.1, 4.2, 0.9, 2.5, 3.6, 4.0, 2.2, 3.3, 1.8])
y_train = np.array([3.6, 2.8, 5.4, 1.9, 2.9, 4.3, 6.2, 8.0, 2.5, 5.0, 7.3, 8.4, 4.7, 6.5, 4.0])

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Biaya iklan dan laba bersih")
plt.xlabel("Biaya iklan perusahaan")
plt.ylabel("Laba bersih perusahaan")
plt.show()


def derivative_of_cost_in_respect_to_w(x, y, weight, bias):
    m = len(x)
    derivative = 0
    for i in range(m):
        f_wb = weight * x[i] + bias
        cost = (f_wb - y[i]) * x[i]
        derivative += cost
    derivative /= m
    return derivative


def derivative_of_cost_in_respect_to_b(x, y, weight, bias):
    m = len(x)
    derivative = 0
    for i in range(m):
        f_wb = weight * x[i] + bias
        cost = (f_wb - y[i])
        derivative += cost
    derivative /= m
    return derivative


w = 0
b = 0
learning_rate = 0.1
not_converged = True

while not_converged:
    d_w = derivative_of_cost_in_respect_to_w(x_train, y_train, w, b)
    d_b = derivative_of_cost_in_respect_to_b(x_train, y_train, w, b)

    temp_w = w - learning_rate * d_w
    temp_b = b - learning_rate * d_b

    if (temp_w == w) and (temp_b == b):
        not_converged = False

    w = temp_w
    b = temp_b


def compute_line(x, weight, bias):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = weight * x[i] + bias
    return f_wb


regression_line = compute_line(x_train, w, b)
plt.plot(x_train, regression_line, c='b', label='prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='reality')
plt.title("Biaya iklan dan laba bersih")
plt.xlabel("Biaya iklan perusahaan")
plt.ylabel("Laba bersih perusahaan")
plt.legend()
plt.show()
