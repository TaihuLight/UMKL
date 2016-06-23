from __future__ import division
import numpy as np
from scipy.linalg import *
import sklearn
import sys

norm = np.linalg.norm

def umkl_descent(kernels, rho, epsilon=0.1, p=10):
    n = kernels[0].shape[0]
    
    # Obtain k_i from eigenvalue decompositions 
    # of given kernels. (Only p largest eigenvalues)
    w, K = eigh(kernels[0], eigvals=(0,p-1))
    for i in range(K.shape[1]):
        K[:,i] *= w[i]
    for kernel in kernels[1:]:
        w, v = eigh(kernel, eigvals=(0,p-1))
        for i in range(v.shape[1]):
            v[:,i] *= w[i]
        K = np.hstack((K, v))

    m = K.shape[1]

    U = np.random.randn(m, n)
    obj_term1 = rho*sum([norm(U[i,:]) for i in range(m)])

    Z = np.eye(n)
    for i in range(m):
        Z -= np.outer(K[:,i], U[i,:])
    
    prev_obj = obj_term1 + norm(Z, 'fro')

    diff = prev_obj
    while abs(diff) > epsilon:
        for i in range(m):
            Z += np.outer(K[:,i], U[i,:])

            # Actual descent
            a = norm(np.dot(K[:,i].T, Z))**2
            b = norm(Z, 'fro')**2
            c = norm(K[:,i])**2
            d = (rho**2) - c
            alpha = ( a*d + np.sqrt( (a*d)**2 - a*c*d*( (rho**2)*b - a ) ) ) / (a*c*d)

            # Descend and update objective value
            temp = obj_term1 - rho*norm(U[i,:])
            U[i,:] = alpha * np.dot(K[:,i].T,  Z)
            temp += rho*norm(U[i,:])
            obj_term1 = temp

            Z -= np.outer(K[:,i], U[i,:])
            new_obj = obj_term1 + norm(Z, 'fro')

        diff = prev_obj - new_obj
        print diff
        prev_obj = new_obj

    Phi = new_obj
    print "Optimal value:", Phi
    weights = [ norm(Z, 'fro')/Phi ]

    for i in range(m):
        weights.append( norm(U[i,:])/Phi )

    optimal_kernel = weights[0]*np.eye(n)
    for i in range(m):
        optimal_kernel += (rho**(-2)) * weights[i] * np.outer(K[:,i], K[:,i])
    
    return optimal_kernel
    

if __name__ == '__main__':
    kernels_file = sys.argv[1]
    kernels = np.load(kernels_file)
    kernels = [k.todense() for k in kernels]
    optimal_kernel = umkl_descent(kernels, 1)
