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
    return math.exp(math.sin(x) + math.cos(x))
    #return 3*math.cos(15*x)


a = 0
b = 2*math.pi
n = 3
step = 0.01



t_j = [(math.pi*j)/n for j in range(2*n)] 
x_interp = get_m_dots(a, b, step)
y_interp = [func(i) for i in x_interp]
y = [func(i) for i in t_j]

a_k = []
for k in range(0, n+1):
    a_k_j = 0
    for j in range(2*n):
        a_k_j += y[j]*math.cos((math.pi*j*k)/n)
    a_k.append(a_k_j/n)


b_k = []
for k in range(1, n):
    b_k_j = 0
    for j in range(2*n):
        b_k_j += y[j]*math.sin((math.pi*j*k)/n)
    b_k.append(b_k_j/n)


def func_g(t, a_k, b_k):
    result = a_k[0]/2 + a_k[-1]/2
    for k in range(1, n):
        result += a_k[k]*math.cos(k*t) + b_k[k-1]*math.sin(k*t)
    return result


y_g = [func_g(i, a_k, b_k) for i in t_j]

plt.plot(x_interp, y_interp)
plt.plot(t_j, y_g)
plt.savefig('graphics.png')

