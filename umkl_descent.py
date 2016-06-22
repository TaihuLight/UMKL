from __future__ import division
import numpy as np
import sklearn
import sys

norm = np.linalg.norm
epsilon = 0.001

def umkl_descent(kernels, rho):
    n = kernels[0].shape[0]
    
    # Obtain k_i from eigenvalue decompositions 
    # of given kernels. TODO: Approximate computation
    w, K = np.linalg.eig(kernels[0])
    for i in range(K.shape[1]):
        K[:,i] *= w[i]
    for kernel in kernels[1:]:
        w, v = np.linalg.eig(kernel)
        for i in range(v.shape[1]):
            v[:,i] *= w[i]
        K = np.hstack((K, v))

    m = K.shape[1]

    # Objective function
    def obj_func(U):
        t1 = rho*sum([norm(U[i,:]) for i in range(m)])
        M = np.eye(n)
        for i in range(m):
            M -= np.outer(K[:,i], U[i,:])
        t2 = norm(M, 'fro')
        return t1 + t2

    U = np.random.randn(m, n)

    Z = np.eye(n)
    for i in range(m):
        Z -= np.outer(K[:,i], U[i,:])

    # TODO: currently checking convergence of objective value,
    # possibly cheaper to check convergence of coordinates
    converged = False
    while not converged:
        for i in range(m):
            Z += np.outer(K[:,i], U[i,:])
            old_obj = obj_func(U)

            # Actual descent
            a = norm(np.dot(K[:,i].T, Z))**2
            b = norm(Z, 'fro')**2
            c = norm(K[:,i])**2
            d = (rho**2) - c
            alpha = ( a*d + np.sqrt( (a*d)**2 - a*c*d*( (rho**2)*b - a ) ) ) / (a*c*d)
            U[i,:] = alpha * np.dot(K[:,i].T,  Z)

            new_obj = obj_func(U)
            diff = old_obj - new_obj
            print diff
            if diff < epsilon:
                converged = True
            else:
                converged = False

            Z -= np.outer(K[:,i], U[i,:])

    
    Phi = obj_func(U)
    weights = [ norm(Z, 'fro')/Phi ]

    for i in range(1,m):
        weights.append( norm(U[i,:])/Phi )

    return weights
    

if __name__ == '__main__':
    kernels_file = sys.argv[1]
    kernels = np.load(kernels_file)
    kernels = [k.todense() for k in kernels]
    weights = umkl_descent(kernels, 0.7)
