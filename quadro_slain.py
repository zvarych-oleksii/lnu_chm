import matplotlib.pyplot as plt


def quadro_beta_spline(beta):
    lst = []
    print(beta)
    for i in range(len(beta)):
        if abs(beta[i]) <= 0.5:
            lst.append(1/2 * (2-(abs(beta[i])-0.5)**2 - (abs(beta[i]) + 0.5)**2))
        elif abs(beta[i])<=1.5 and abs(beta[i])>=0.5:
            lst.append(1/2 * ((abs(beta[i])-1.5)**2))
        else:
            lst.append(0)
    print(lst)
    return lst

def quadro_spline():
    a = 0
    b = 10
    n = 10
    N = 100
    h = (b - a) / n
    h_1 = (b - a) / N
    x_k = [a + (k * h) for k in range(0, n + 1)]
    x_values = [a + (i * h_1) for i in range(N+1)]
    for k in range(n+1):
        beta = [(x_values[i] - x_k[k]) / h for i in range(N+1)]
        plt.plot(x_values, quadro_beta_spline(beta), label='Quadro Beta-Spline')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Quadro Beta-Spline Function')
        plt.legend()
        plt.grid(True)
    plt.savefig('quadro_beta_spline.png')

quadro_spline()
