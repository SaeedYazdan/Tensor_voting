import numpy as np
from numpy import linalg as LA
import math



def compute_stick_tf(v, N='', n='', sigma='', c=''):

    # compute_stick_tf - compute the voting kernel of a stick tensor.
    #
    #   T = compute_stick_tf(v,n,N,sigma,c);
    #
    #   'v' is the non-null eigenvector of the stick tensor (i.e. the tensor is v*v').
    #   'N' is the total size of the image (supposed to lie in [0,1]x[0,1]).
    #
    #       The equation for the decay of the field is
    #           DF(s,k) = exp(-(s^2+c*k^2)/sigma^2)
    #       where 's' is the length of the circle that joins
    #       the two points, and 'k' is its curvature.
    #
    #   Optional:
    #   'sigma' control the scale of the voting field.
    #   'c' control the ratio between distance and curvature.
    #       The higher 'c' is, the narrower the field will be.
    #   'n' is the size of the kernel (should be odd).
    #
    #   See also compute_stick_ball.
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if N == '':
        raise Exception('Not enough arguments')
    
    if n == '':
        n = N / 2
    
    if sigma == '':
        sigma = 0.2
    
    if c == '':
        c = 0.1 * sigma

    # n should be odd
    n = int(n / 2) * 2 + 1
    nn = (n - 1) / 2
    # v should be of unit length
    v = v / LA.norm(v)
    # v = v(:);
    # w = [-v(2);v(1)]   # normal vector
    w = [-v[1], v[0]]

    # x = [-nn:nn] / N
    x0 = list()
    #x0.append(-nn)
    n0 = -nn
    while n0 <= nn:
        x0.append(n0/N)
        n0 += 1
    [Y, X] = np.meshgrid(x0, x0)

    # rotate the grid to align v to [1;0], ie. multiply by [v,w]'
    #M = [X, Y].T
    M = np.concatenate((np.reshape(Y, (1, -1)).T, np.reshape(X, (1, -1)).T), axis=1).T
    # A = [v, w]
    A = np.stack((v, w)).T
    # M = (A.T) * M
    M = np.dot((A.T), M)
    # X = np.reshape(M[0, :], [n, n])
    X = M[0, :].reshape((n, n)).T
    #Y = np.reshape(M[1, :], [n, n])
    Y = M[1, :].reshape((n, n)).T

    # angle
    #theta = math.atan2(Y, X)
    theta = np.zeros((n, n))
    thetas = np.zeros((n, n))

    for i in range(len(X)):
        for j in range(len(Y)):
            theta[i, j] = math.atan2(Y[i, j], X[i, j])
            
            thetas[i, j] = abs(theta[i, j])
            if thetas[i, j] > (math.pi / 2):
                thetas[i, j] = math.pi - thetas[i, j]
            
    #I = find(thetas > pi / 2)
    #I = np.where(thetas > math.pi/2)
    #thetas[I] = math.pi - thetas[I]
    # length
    #L = math.sqrt(X ** 2 + Y ** 2)
    # arc-length
    L = np.zeros((n, n))
    s = np.zeros((n, n))
    k = np.zeros((n, n))
    
    for i in range(len(X)):
        for j in range(len(Y)):
            temp = math.sqrt(X[i, j] ** 2 + Y[i, j] ** 2)
            L[i, j] = temp
            if temp != 0 and thetas[i, j] != 0:
                s[i, j] = temp * thetas[i, j] / math.sin(thetas[i, j])
            elif temp == 0 or thetas[i, j] == 0:
                s[i, j] = temp
            if temp != 0:
                k[i, j] = 2 * math.sin(thetas[i, j]) / temp
    #I = find(L != 0 and thetas != 0)
    #s[I] = L[I] * thetas[I] / math.sin(thetas[I])
    #I = find(L == 0 or thetas == 0)
    #s[I] = L[I]
    # curvature
    
    #I = find(L != 0)
    #k[I] = 2 * math.sin(thetas[I]) / L[I]
    # attenuation
    DF = np.exp(-(s ** 2 + c ** 2 * k ** 2) / sigma ** 2) # ???????????????????????????????
    

    # the direction vector is a = A*[-cos(2*theta);-sin(theta)]
    # and the tensor is DF*a*a'
    b = np.zeros((n, n, 2))
    a = np.zeros((n, n, 2))
    #b[:, :, 0] = -math.cos(2 * theta)
    #b[:, :, 1] = -math.sin(2 * theta)

    for i in range(n):
        for j in range(n):
            b[i, j, 0] = -math.cos(2* theta[i, j])
            b[i, j, 1] = -math.sin(2* theta[i, j])
            a[i, j, 0] = A[0, 0] * b[i, j, 0] + A[0, 1] * b[i, j, 1]
            a[i, j, 1] = A[1, 0] * b[i, j, 0] + A[1, 1] * b[i, j, 1]
                
    # rotate point wise
    #a = b
    #a[:, :, 0] = A[0, 0] * b[:, :, 0] + A[0, 1] * b[:, :, 1]
    #a[:, :, 1] = A[1, 0] * b[:, :, 0] + A[1, 1] * b[:, :, 1]
    # compute rigidity tensor associated to a
    T = np.zeros((n, n, 2, 2))
    
    T[:, :, 0, 0] = DF * (a[:, :, 0] ** 2)
    T[:, :, 0, 1] = DF * a[:, :, 0] * a[:, :, 1]
    T[:, :, 1, 0] = T[:, :, 0, 1]
    T[:, :, 1, 1] = DF * a[:, :, 1] ** 2

    return T
