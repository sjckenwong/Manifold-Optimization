import numpy as np
from scipy.optimize import minimize

def steepStiefel(f, Y0, epsilon=1e-6):
    Y = Y0/2
    Ynew = Y0

    while np.linalg.norm(Y-Ynew) > epsilon:
        Y = Ynew
        G = f(Y) - Y*Y.tranpose()*f(Y)
        U, s, V = np.linalg.svd(-G, full_matrices=False)
        sigma = np.diag(s)
        fun = lambda t: f(Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose())
        res = minimize(fun, 0)
        t = res.x
        Ynew = Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose()

    return Ynew

def newtonStiefel(f, Y0, epsilon=1e-6):
    Y= Y0/2
    Ynew = Y0

    while np.linalg.norm(Y-Ynew) > epsilon:
        Y = Ynew
        G = f(Y) - Y*Y.tranpose()*f(Y)
        # solve delta
        U, s, V = np.linalg.svd(delta, full_matrices=False)
        sigma = np.diag(s)
        fun = lambda t: f(Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose())
        res = minimize(fun, 0)
        t = res.x
        Ynew = Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose()

    return Ynew

def conjugStiefel(f, Y0, epsilon=1e-6):
    Y = Y0/2
    Ynew= Y0
    n, k = Y.shape
    modTerm = (k+1)*(n-k)
    i = 0

    while np.linalg.norm(Y-Ynew) > epsilon:
        i += 1
        Y = Ynew
        G = f(Y) - Y*Y.transpose()*f(Y)
        H = -G
        U, s, V = np.linalg.svd(H, full_matrices=False)
        sigma = np.diag(s)
        fun = lambda t: f(Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose())
        res = minimize(fun, 0)
        t = res.x
        Ynew = Y*V*np.cos(t*sigma)*V.transpose() + U*np.sin(t*sigma)*V.tranpose()
        Gnew = f(Ynew) - Ynew*Ynew.transpose()*f(Ynew)
        tauH = (-Y*V*np.sin(t*sigma) + U*np.cos(t*sigma)) * sigma * V.transpose()
        tauG = G - (Y*V*np.sin(t*sigma) + U*(np.eye(len(sigma))-np.cos(t*sigma))*U.transpose()*G)
        gamma = np.trace((Gnew-tauG).transpose()*Gnew) / np.trace(G.transpose()*G)
        Hnew = -Gnew + gamma*tauH
        if (i+1) % modTerm == 0:
            Hnew = -Gnew

    return Ynew

def steepProj(f, P0, epsilon=1e-6):
    n = len(P0)
    P = P0/2
    Pnew = P0

    while np.linalg.norm(P-Pnew) > epsilon:
        P = Pnew
        fP = f(P)
        gradf = (P*(P*fP-fP*P)-(P*fP-fP*P)*P)
        #solve delta and Z
        A = 0.5*np.cos(2*np.sqrt(Z*Z.transpose()))
        B = -np.sinc(2*np.sqrt(Z*Z.transpose()))*Z
        C = B.transpose()
        D = 0.5*np.sin(2*np.sqrt(Z.transpose()*Z))
        Pnew = 0.5*np.eye(n) + delta.transpose()*np.bmat([[A, B], [C, D]])*delta

    return Pnew

def newtonProj(f, P0, epsilon=1e-6):
    n = len(P0)
    P = P0/2
    Pnew = P0

    while np.linalg.norm(P-Pnew) > epsilon:
        P = Pnew
        #solve omega
        #solve delta
        Q, R = np.linalg.qr(theta*(np.eye(n)-t*(P*(P*omega-omega*P)-(P*omega-omega*P)*P))*theta.transpose())
        Pnew = theta.transpose()*Q*theta*P*theta.transpose()*Q.transpose()*theta

    return Pnew

