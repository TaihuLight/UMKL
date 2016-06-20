from __future__ import division
import numpy as np
import sklearn
import sys

norm = np.linalg.norm
epsilon = 0.01

def umkl_descent(kernels, rho):
    # Initialize y
    n = kernels[0].shape[0]
    y = np.eye(n)
    
    # Obtain k_i from eigenvalue decompositions 
    # of given kernels. TODO: Sparse or approximate computation?
    w, K = np.linalg.eig(kernels[0].todense())
    for kernel in kernels[1:]:
        w, v = np.linalg.eig(kernel.todense())
        K = np.hstack((K, v))

    m = K.shape[1]

    # Objective function
    def obj_func(U):
        t1 = rho*sum([norm(U[i,:]) for i in range(m)])
        M = y
        for i in range(m):
            M -= np.outer(K[:,i], U[i,:])
        t2 = norm(M, 'fro')
        return t1 + t2

    Z = y
    for i in range(m):
        Z -= np.outer(K[:,i], U[i,:])

    U = np.random.randn(m, n)
    prev_obj = obj_func(U)
    diff = prev_obj

    # TODO: currently checking convergence of objective value,
    # possibly cheaper to check convergence of coordinates
    while abs(diff) > epsilon:
        for i in range(m):
            Z += np.outer(K[:,i], U[i,:])

            a = norm(Z.T * K[:,i])**2
            b = norm(Z, 'fro')**2
            c = norm(Z.T * np.outer(K[:,i], K[:,i]), 'fro')**2
            d = (rho**2) * a - c

            alpha = ( a*d + np.sqrt( (a*d)**2 - a*c*d*( (rho**2)*b - a ) ) ) / (c*d)

            U[i,:] = alpha * K[:,i].T * Z
            Z -= np.outer(K[:,i], U[i,:])

        new_obj = obj_func(U)
        diff = prev_obj - new_obj
        prev_obj = new_obj
    
    Phi = new_obj
    weights = [ norm(Z, 'fro')/Phi ]

    for i in range(1,m):
        weights.append( norm(U[i,:])/Phi )

    return weights
    

if __name__ == '__main__':
    kernels_file = sys.argv[1]
    kernels = np.load(kernels_file)
    weights = umkl_descent(kernels, 0.1)
