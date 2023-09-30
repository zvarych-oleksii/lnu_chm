import numpy as np
from matplotlib import pyplot as plt 
import math


def get_m_dots(a,b, n):
    point_list = [i for i in range(int(a*1000), int(b*1000)+1, int(step*1000))]
    point_list = [x/1000 for x in point_list]
    return point_list


def get_jeb_dots(a, b, n):
    point_list = []
    for i in range(n+1):
        point_list.append((a+b)/2+(b-a)/2 * math.cos(((2*i+1)/(2*(n+1))*math.pi)))
    return point_list


def func(x):
    return (1)/(1+25*(x**2))
    #return math.log(x+2) 


def lagrange_func(x, y, x_interp):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i!=j:
                term *= (x_interp - x[j]) / (x[i] - x[j])
        result += term
    return result



def divided_difference(x_values, y_values):
    n = len(x_values)
    table = [[0] * n for _ in range(n)]
    
    for i in range(n):
        table[i][0] = y_values[i]
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_values[i + j] - x_values[i])
    
    return [table[0][j] for j in range(n)]


def newton_func(x_values, y_values, x):
    n = len(x_values)
    coefficients = divided_difference(x_values, y_values)
    result = coefficients[0]
    product_term = 1
    
    for i in range(1, n):
        product_term *= (x - x_values[i - 1])
        result += coefficients[i] * product_term
    
    return result


a = -1
b = 1
n = 5
step = 0.001


def get_dots_for_func(a,b, n, cheb = False):
    if cheb:
        x = get_jeb_dots(a, b, n)
        y = [func(x_i) for x_i in x_cheb]
    else:
        h = (b-a)/n
        x = [(a+i*h) for i in range(n+1)]
        y = [func(x_i) for x_i in x]
    return x,y



x, y = get_dots_for_func(a, b, n)

x_interp = get_m_dots(a, b, step)
y_lagrange = [lagrange_func(x, y, x_i) for x_i in x_interp]
y_default = [func(x_i) for x_i in x_interp]
y_newton = [newton_func(x, y, x_i) for x_i in x_interp]

plt.title("Plot test")
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.plot(x_interp, y_newton)
plt.plot(x_interp, y_default)
plt.plot(x_interp, y_lagrange)
plt.legend(['Newton func', 'Default func', 'Lagrange func'])

plt.savefig('graphics.png')



def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("Input lists must have the same length")
    
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors) / n
    return mse

# Calculate MSE for y_lagrange
mse_lagrange = mean_squared_error(y_default, y_lagrange)

# Calculate MSE for y_newton
mse_newton = mean_squared_error(y_default, y_newton)

print(f"Mean Squared Error for Lagrange Interpolation: {mse_lagrange}")
print(f"Mean Squared Error for Newton Interpolation: {mse_newton}")
