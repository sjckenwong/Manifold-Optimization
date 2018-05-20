import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr

def steepGrass(f, A, Y0, epsilon=1e-3):
    Y = Y0/2
    Ynew = Y0
    n, p = Y0.shape

    while np.linalg.norm(Y-Ynew) > epsilon:
        Y = Ynew
        G = (np.eye(n) - Y*Y.T) * (A*Y - Y*Y.T*A*Y)     # gradient of f on Gr(5,3)
        U, s, V = np.linalg.svd(-G, full_matrices=False)   # compact SVD
        sigma = np.diag(s)
        fun = lambda t: f(Y*V*np.cos(t*sigma)*V.T + U*np.sin(t*sigma)*V.T)
        res = minimize(fun, 1) # exact line search on geodesic
        t = res.x
        Ynew = Y*V*np.cos(t*sigma)*V.T + U*np.sin(t*sigma)*V.T

    return Ynew

if __name__ == '__main__':

    # random 5 x 5 symetric matrix A
    b = 10*np.random.rand(5,5)
    A = (b + b.T)/2

    # Optimization on Gr(5,3). Minimize f(Y) = 0.5*tr(Y'AY)
    f = lambda Y: 0.5*np.trace(Y.T*A*Y)

    # random initalization Y0
    Y0, _ = qr(np.random.randn(5, 5))
    Y0 = np.matrix(Y0[:,:3])

    res = steepGrass(f, A, Y0)

    return res


