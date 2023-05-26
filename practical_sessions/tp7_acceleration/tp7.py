import numpy as np
import matplotlib.pyplot as plt

def x(k):
    s = 0
    for i in range(k):
        s += ((-1)**i) / (2*i + 1)
    return 4*s

def a(k):
    return (x(k+1) - x(k+2)) / (x(k) - x(k+1))

def b(k):
    return x(k+1) - a(k)*x(k)

