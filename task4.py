import numpy as np
import matplotlib.pyplot as plt

def best_fit_polynomial(x, y, degree):
    # Знаходимо коефіцієнти полінома найкращого середньоквадратичного наближення
    coefficients = np.polyfit(x, y, degree)

    # Побудова полінома
    polynomial = np.poly1d(coefficients)

    return polynomial

def plot_approximation(f, x_range, degree, title):
    # Генерація точок для функції f
    x = np.linspace(x_range[0], x_range[1], 100)
    y = f(x)

    # Додавання шуму до значень функції для реалістичності
    y += np.random.normal(0, 0.1, len(x))

    # Знаходження полінома найкращого середньоквадратичного наближення
    polynomial = best_fit_polynomial(x, y, degree)

    # Побудова графіків
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Noisy Data')
    plt.plot(x, f(x), label='True Function', color='green')
    plt.plot(x, polynomial(x), label=f'Approximation (Degree {degree})', color='red')
    plt.title(title)
    plt.legend()
    plt.savefig('best_approximation.png')

# Задані функції
def f1(x):
    return x**2

def f2(x):
    return 3 * x**3 - 1

# Задані відрізки
x_range1 = [-1, 2]
x_range2 = [-1, 1]

# Степінь полінома найкращого середньоквадратичного наближення
degree = 2

# Побудова наближення для f(x) = x^2
plot_approximation(f1, x_range1, degree, 'Approximation for f(x) = x^2')

# Побудова наближення для f(x) = 3x^3 - 1
#plot_approximation(f2, x_range2, degree, 'Approximation for f(x) = 3x^3 - 1')
