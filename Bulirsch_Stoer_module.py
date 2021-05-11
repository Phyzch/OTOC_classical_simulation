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

        Result = 0.5*(y1 + y0 + h*F(y1, x , *args))
        dof = int( len(Result ) / 2)
        for i in range(dof):
            if(Result[i] < 0):
                Result[i] = 0

        return Result

    def richardson(r, k):
        # https://www.wikiwand.com/en/Richardson_extrapolation
        # t = (k/k-1) due to time step we choose for h/N is N = 2 * k.
        # after each step, r[1], r[2], r[3] are at different order approximation to A*.  r[1] is most accurate.
        for j in range(k-1,0,-1):
            const = (k/(k - 1.0))**(2.0*(k-j))
            r[j] = (const*r[j+1] - r[j])/(const - 1.0)
        return

    Finish_simulation = True

    kMax = 101
    n = len(y)
    r = zeros((kMax, n), dtype=float)

    # Start with two integration steps
    nSteps = 2
    r[1] = midpoint(F, x, y, xStop, nSteps , args)
    r_old = r[1].copy()

    # Increase the number of integration points by 2 and refine result by Richardson extrapolation
    e = 0
    e_old = 10000
    for k in range(2,kMax):
        nSteps = 2*k
        r[k] = midpoint(F, x, y, xStop, nSteps , args)
        richardson(r,k)

        # Compute RMS change in solution
        e = math.sqrt(sum((r[1] - r_old)**2)/n)

        if( abs(e) > 2 * abs(e_old)):
            if(abs(e_old) > 1):
                print("May have divergence problem. ")
                print("e old: " + str(e_old) + "  e:  " + str(e))
                Finish_simulation = False

            return r_old , Finish_simulation

        # Check for convergence
        if e < tol:
            return r[1] , Finish_simulation

        r_old = r[1].copy()
        e_old = e

    print("error now:  " + str(e))
    print('action:  ' + str(r[1]))
    print("Midpoint method did not converge")
    return r[1], Finish_simulation

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

    finish_simulation = True

    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:

        H = min(H,xStop - x)
        y, finish_simulation_this_time = integrate_BulirschStoeir(F, x, y, x + H, tol, args)   # Midpoint method
        x = x + H

        if(  finish_simulation_this_time == False ):
            print("no return value. Now break cycle.  t =  " + str(x))
            finish_simulation = False
            break
        X.append(x)
        Y.append(y)
    return array(X), array(Y) , finish_simulation

