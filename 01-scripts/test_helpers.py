import helpers
import numpy as np
import matplotlib.pyplot as plt

"""
BEGIN FINITE DIFFERENCE TEST

Test whether rk4 returns the correct solution to

    dy
   ----- = sin(x), where y(0) = 1.
    dx

The analytical solution is 2-cos(x).
"""
Δx = 0.1
xvals = np.arange(0.0,10.0,Δx)

def dydx(x, y):
    return np.sin(x)

solution = [2-np.cos(x) for x in xvals]

yvals = np.zeros(len(xvals))
yvals[0] = 1.0

for n in range(len(yvals)-1):
    yvals[n+1] = helpers.rk4(dydx, xvals[n], yvals[n], Δx)

plt.figure(figsize=(8, 8))
plt.plot(xvals, yvals, 'r*', label="Numerical")
plt.plot(xvals, solution, 'k--', lw=2, label="Exact")
plt.legend(loc=0)
plt.xlabel("y(x)")
plt.ylabel("x")
plt.show()
"""
END RUNGE-KUTTA TEST
"""
