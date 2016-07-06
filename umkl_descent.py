# The function umkl_descent implements a coordinate descent
# algorithm for unsupervised multiple-kernel learning.
# The main script should be run with a .npy file as the
# command line argument. The .npy file should contain the
# multiple kernels as a collection of sparse scipy matrices.

from __future__ import division
import numpy as np
from scipy.linalg import *
import sklearn
import sys

norm = np.linalg.norm

def umkl_descent(kernels, rho, epsilon=0.001, p=10):
    # Obtain k_i from eigenvalue decompositions 
    # of given kernels. (Only p largest eigenvalues)
    n = kernels[0].shape[0]
    print 'Peforming UMKL for ' + str(n) + ' X ' + str(n) + ' kernels.'
    q = kernels[0].shape[1]
    w, K = eigh(kernels[0], eigvals=(q-p,q-1))
    for i in range(K.shape[1]):
        K[:,i] *= np.sqrt(w[i])
    for kernel in kernels[1:]:
        w, v = eigh(kernel, eigvals=(q-p,q-1))
        for i in range(v.shape[1]):
            v[:,i] *= np.sqrt(w[i])
        K = np.hstack((K, v))

    # Make sure rho is small enough
    m = K.shape[1]
    min_k_norm = min([norm(K[:,i]) for i in range(m)])
    rho = min(rho, 0.5*min_k_norm)

    # Initialize U and compute initial objective value
    U = np.random.randn(m, n)
    obj_term1 = rho*sum([norm(U[i,:]) for i in range(m)])

    Z = np.eye(n)
    for i in range(m):
        Z -= np.outer(K[:,i], U[i,:])
    
    prev_obj = obj_term1 + norm(Z, 'fro')

    # Descent loop
    objective_values = [prev_obj]
    diff = prev_obj
    while diff > epsilon:
        for i in range(m):
            # Update Z 
            Z += np.outer(K[:,i], U[i,:])

            # Actual descent
            a = norm(np.dot(K[:,i].T, Z))**2
            b = norm(Z, 'fro')**2
            c = norm(K[:,i])**2
            d = (rho**2) - c
            alpha = ( a*d + np.sqrt( (a*d)**2 - a*c*d*( (rho**2)*b - a ) ) ) / (a*c*d)
            if np.isnan(alpha):
                print "Alpha imaginary."
                exit(0)
            if alpha < 0:
                print "Alpha negative."
                exit(0)

            # Descend and update objective value
            temp = obj_term1 - rho*norm(U[i,:])
            U[i,:] = alpha * np.dot(K[:,i].T,  Z)
            temp += rho*norm(U[i,:])
            obj_term1 = temp

            Z -= np.outer(K[:,i], U[i,:])
            new_obj = obj_term1 + norm(Z, 'fro')
            objective_values.append(new_obj)

        # Check convergence
        diff = prev_obj - new_obj
        prev_obj = new_obj

        if diff < 0:
            print "Objective value increased."
            exit(0)

    # Recover primal variables: optimal weights
    Phi = new_obj
    print "Optimal value:", Phi
    weights = [ norm(Z, 'fro')/Phi ]

    for i in range(m):
        weights.append( rho*norm(U[i,:])/Phi )

    # Compute optimal kernel
    #optimal_kernel = weights[0]*np.eye(n)
    #for i in range(m):
    #    optimal_kernel += (rho**(-2)) * weights[i] * np.outer(K[:,i], K[:,i])
    
    return weights, objective_values
    

if __name__ == '__main__':
    kernels_file = sys.argv[1]
    kernels = np.load(kernels_file)
    kernels = [k.todense() for k in kernels]
    weights, objective_values = umkl_descent(kernels, 0.2)
