import numpy as np

import compute_ball_tf as cbt
import perform_convolution as con



def perform_voting_ball(M, sigma='', c='', p='', n=''):

    # perform_voting - perform the voting process for the ball field.
    #
    #   T = perform_voting_ball(M,sigma,p,n);
    #
    #   'M' is a 2D function made by the user (1 for features and 0 in the background).
    #   
    #   Optional:
    #   'sigma' and 'c' control the stick field (see 'compute_stick_tf').
    #   'n' is the size of the kernel (typically smaller than the size of the image).
    #   'p' is the precision for the computation of the ball tensor field.
    #
    #   See also perform_voting.
    #
    #   Copyright (c) 2004 Gabriel Peyré


    if sigma == '':
        sigma = 0.2
    if c == '':
        c = 0.1 * sigma
    if p == '':
        p = 32

    N = len(M)
    n = N / 2
    n = int(n / 2) * 2 + 1 # should be odd

    # compute the ball voting field
    B = cbt.compute_ball_tf(N, n, sigma, c, p)

    T = np.zeros((M.shape[0], M.shape[1], 2, 2))
    for i in range(2):
        for j in range(2):
            # T(:,:,i) = conv2(M,B(:,:,i),'same');              # use inplace acyclic convolution
            T[:, :, i, j] = con.perform_convolution(M, B[:, :, i, j])              # use inplace acyclic convolution
            # T(:,:,i) = perform_convolution(M,kernel(:,:,i));   # use inplace symmetric convolution
    
    return T
