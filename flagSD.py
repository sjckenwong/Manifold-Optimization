import numpy as np
from scipy.linalg import qr, expm


def flagSD(n, m):

    n_length = len(n)-1

    A = np.random.random((m,m))
    A = 0.5 * (A+A.T)

    X0, _ = qr(np.random.random((m,m)))

    X = X0
    fX = np.trace(np.dot(X[:,:n[-1]].T, np.dot(A, X[:,:n[-1]])))


    iter = 0
    norm_RG = 1
    dist = 1

    while iter < 1000 and norm_RG > 1e-6 and dist > 1e-5:
        EG = 2 * np.dot(A, X)
        RG = np.zeros((m, n[-1]))

        # sum_{j=1}^{k} Aj*Dj'
        sumAjDj = np.zeros((m, m))
        for i in range(n_length):
            Ai = X[:, n[i]+1:n[i+1]]
            Di = EG[:, n[i]+1:n[i+1]]
            sumAjDj = sumAjDj + np.dot(Ai, Di.T)

        # compute Riemannian gradient using equation (9)
        for i in range(n_length):
            Ai = X[:, n[i]+1:n[i+1]]
            Di = EG[:, n[i]+1:n[i+1]]
            RG[:, n[i]+1:n[i+1]] = Di - (np.dot(Ai, np.dot(Ai.T, Di)) + np.dot(sumAjDj - np.dot(Ai, Di.T), Ai))

        B = np.zeros((m, m))
        B[:, :n[-1]] = np.dot(X.T, RG)

        B[:n[-1], n[-1]:m] = -B[n[-1]:m, :n[-1]].T

        t = 1
        beta = 0.5
        Xnew = np.dot(X, expm(t*B))

        while t > 1e-6 and np.trace(np.dot(Xnew[:, :n[-1]].T, np.dot(A, Xnew[:, :n[-1]]))) < fX:
            t *= beta
            Xnew = np.dot(X, expm(t*B))

        Xnew, _ = qr(Xnew)

        norm_RG = np.linalg.norm(RG, 'fro')
        dist = distance(X[:, :n[-1]], )
        iter += 1

        X = Xnew
        fX = np.trace(np.dot(X[:, :n[-1]].T, np.dot(A, X[:, :n[-1]])))

    return X



