import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

def build_plot(x_values, splines):
    for spline in splines:
        plt.plot(x_values, spline, label='Linear Beta-Spline')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Beta-Spline Function')
        plt.legend()
        plt.grid(True)
    plt.savefig('linear_beta_spline.png')


def get_func_nodes(x_values):
    return [np.sin(x) for x in x_values]


def linear_beta(beta):
    if abs(beta) <= 1:
        return 1-abs(beta)
    else:
        return 0


def cubic_beta(x):
    if abs(x) <= 1:
        return (1.0/6) * (((2 - abs(x)) ** 3) - (4 * ((1 - abs(x)) ** 3)))
    elif abs(x) <= 2:
        return (1.0/6) * ((2 - (abs(x))) ** 3)
    else:
        return 0
    

def linear_beta_spline(beta):
    if abs(beta) <= 1:
        return 1-abs(beta)
    else:
        return 0


def get_nodes(n, a, b):
    h = (b - a)/n 
    nodes = [a + (k*h) for k in range(n+1)]
    return nodes


def get_beta(x_k, x_values, h):
    N = len(x_values)
    beta = [(x_values[i] - x_k) / h for i in range(N)]
    return beta


def linear_spline(a, b, n, N):
    h = (b - a) / n
    x_k = get_nodes(n, a, b)
    x_values = get_nodes(N, a, b)
    y_values = get_func_nodes(x_values)
    y_values_k = get_func_nodes(x_k)
    y_for_plot = []
    for x in range(0, N+1):
        beta_y = []
        for i in range(0, n+1):
            beta_y.append(y_values_k[i]*linear_beta_spline((x_values[x]-x_k[i])/h))
        y_for_plot.append(sum(beta_y))
    plt.plot(x_values, y_for_plot)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Beta-Spline Function')
    plt.grid(True)
    plt.savefig('spline.png')

def cube_spline(a, b, n, N):
    h = (b - a) / n

    x_k = get_nodes(n, a, b)
    x_values = get_nodes(N, a, b)
    y_values = get_func_nodes(x_values)
    y_values_k = get_func_nodes(x_k)
    x_k = [a+(-1)*h] + get_nodes(n, a, b)+ [b+(n+1)*h]

    x = sp.symbols('x')
    f = sp.sin(x)
    derivative = sp.diff(f, x)

    a_1 = h*derivative.subs(x, a)
    b_1 = h*derivative.subs(x, b)

    b_for_matrix = [a_1] + y_values_k + [b_1]
    zero_matrix = [[0 for k in range(0, n+3)] for i in range(0, n+3)]
    zero_matrix[0][0] = -0.5
    zero_matrix[0][2] = 0.5
    zero_matrix[-1][-1] = 0.5
    zero_matrix[-1][-3] = -0.5

    for i in range(1, len(zero_matrix)-1):
        zero_matrix[i][i-1] = 1/6
        zero_matrix[i][i] = 2/3
        zero_matrix[i][i+1] = 1/6

    for i in range(len(zero_matrix)):
        print(zero_matrix[i], b_for_matrix[i])

    coefficients = np.array(zero_matrix)
    constants = np.array(b_for_matrix)
    coefficients = coefficients.astype(float)
    constants = constants.astype(float)
    solution = np.linalg.solve(coefficients, constants)

    print(solution)
    y_for_plot = []

    for x in range(0, N+1):
        beta_y = []
        for i in range(0, n+3):
            beta_y.append(solution[i]*cubic_beta((x_values[x]-x_k[i])/h))
        y_for_plot.append(sum(beta_y))

    plt.plot(x_values, y_for_plot)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Beta-Spline Function')
    plt.grid(True)
    plt.savefig('spline.png')
    



a = -2
b = 2
n = 15
N = 100
linear_spline(a, b, n, N) 
cube_spline(a,b,n,N)
