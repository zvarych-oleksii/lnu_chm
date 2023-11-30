import matplotlib.pyplot as plt


def get_func_nodes(x_values):
    return [x**2 for x in x_values]


def linear_beta(beta):
    if abs(beta) <= 1:
        return 1-abs(beta)
    else:
        return 0
    

def linear_beta_spline(beta):
    lst = []
    print(beta)
    for i in range(len(beta)):
        if abs(beta[i]) <= 1:
            lst.append(1-abs(beta[i]))
        else:
            lst.append(0)
    print(lst)
    return lst


def get_nodes(n, a, b):
    h = (b - a)/n 
    nodes = [a + (k*h) for k in range(n+1)]
    return nodes

def get_beta(x_k, x_values, h):
    N = len(x_values)
    beta = [(x_values[i] - x_k) / h for i in range(N)]
    return beta


def linear_spline():
    a = -10
    b = 10
    n = 10
    N = 100
    h = (b - a) / n
    h_1 = (b - a) / N
    x_k = get_nodes(n, a, b)
    x_values = get_nodes(N, a, b)
    for k in range(n+1):
        beta = get_beta(x_k[k], x_values, h)
        linear_spline = [linear_beta(beta[i]) for i in range(len(beta))]
        print(len(linear_spline))
        plt.plot(x_values, linear_spline, label='Linear Beta-Spline')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Beta-Spline Function')
        plt.legend()
        plt.grid(True)
    plt.savefig('linear_beta_spline.png')
linear_spline()



