from numpy import zeros, float, sum
import math
import numpy as np

def integrate_BulirschStoeir(F, x, y, xStop, tol , args):
    # Both return and input should be numpy array for function F.
    def midpoint(F, x, y, xStop, nSteps ,args ):
        # nstep midpoint function.
        h = (xStop - x)/ nSteps
        y0 = y
        y1 = y0 + h*F(y0, x , * args)
        for i in range(nSteps-1):
            x = x + h
            y2 = y0 + 2.0*h*F(y1, x , *args)

            y0 = y1
            y1 = y2

        return 0.5*(y1 + y0 + h*F(y1, x , *args))

    def richardson(r, k):
        # https://www.wikiwand.com/en/Richardson_extrapolation
        # t = (k/k-1) due to time step we choose for h/N is N = 2 * k.
        # after each step, r[1], r[2], r[3] are at different order approximation to A*.  r[1] is most accurate.
        for j in range(k-1,0,-1):
            const = (k/(k - 1.0))**(2.0*(k-j))
            r[j] = (const*r[j+1] - r[j])/(const - 1.0)
        return


    kMax = 101
    n = len(y)
    r = zeros((kMax, n), dtype=float)

    # Start with two integration steps
    nSteps = 2
    r[1] = midpoint(F, x, y, xStop, nSteps , args)
    r_old = r[1].copy()

    # Increase the number of integration points by 2 and refine result by Richardson extrapolation
    for k in range(2,kMax):
        nSteps = 2*k
        r[k] = midpoint(F, x, y, xStop, nSteps , args)
        richardson(r,k)

        # Compute RMS change in solution
        e = math.sqrt(sum((r[1] - r_old)**2)/n)

        # Check for convergence
        if e < tol:
            return r[1]
        r_old = r[1].copy()

    print("Midpoint method did not converge")

# Bulirsch-Stoer Algorithm:-

from numpy import array

def bulStoer(F, x, y, xStop, H, args, tol=1.0e-6):
    ''' X, Y = bulStoer(F, x, y, xStop, H, tol=1.0e-6).
        Simplified Bulirsch-Stoer method for solving the
        initial value problem {y}’ = {F(x,{y})}, where {y} = {y[0],y[1],...y[n-1]}
        x, y = initial conditions
        xStop = terminal value of x
        H = increment of x at which results are stored
        F = user-supplied function that returns the array F(x,y) = {y’[0],y’[1],...,y’[n-1]} '''

    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:

        H = min(H,xStop - x)
        y = integrate_BulirschStoeir(F, x, y, x + H, tol, args)   # Midpoint method
        x = x + H

        if(  type(y)!= np.ndarray ):
            print("no return value. Now break cycle.  t =  " + str(x))
            break
        X.append(x)
        Y.append(y)
    return array(X), array(Y)

