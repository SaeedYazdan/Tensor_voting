import numpy as np

import perform_tensor_decomp as ptd
import perform_voting_ball as pvb
import compute_stick_tf as cst



def perform_voting(M, sigma='', c='', n='', p='', thresh=''):

    # perform_voting - perform the voting process.
    #
    #   T = perform_voting(M,sigma,c,n,p,thresh);
    #
    #   'M' is a tensor field created by the user.
    #
    #   - The oriented features of the image must be stored
    #     in the *highest* eigenvectors of the tensors.
    #   - The confidence in these featres is reflected by
    #     ratio of the eigenvalues 
    #     (l1/l2==1 imply no confidence at all, l2=0 imply total confidence).
    #
    #   Optional:
    #   'sigma' and 'c' control the stick field (see 'compute_stick_tf').
    #   'n' is the size of the kernel (typically smaller than the size of the image).
    #   'p' is the precision for the computation of the ball tensor field.
    #   'thresh' is a threshold (perform voting only if l1-l2>thresh where (l1,l2) are the eigenvalues).
    #
    #   See also perform_voting_ball.
    #
    #   Copyright (c) 2004 Gabriel Peyré

    verb = 1

    # N = size(M, 1)
    N = M.shape[0]
    
    if sigma == '':
        sigma = 0.2
    if c == '':
        c = 0.1 * sigma
    if n == '':
        n = N / 2
    if p == '':
        p = 32
    
    n = int(n / 2) * 2 + 1
    nn = (n - 1) / 2

    # perform eigen-decomposition
    [e1, e2, l1, l2] = ptd.perform_tensor_decomp(M)
        
    # perform ball voting
    T = pvb.perform_voting_ball(l2, sigma, c)

    # perform a threshold to use only high coef
    if thresh == '':
        thresh = np.mean(np.mean((l1 - l2))) / 10
    #I = find((l1 - l2) > thresh)
    I = np.where((l1 - l2) > thresh)
    I = np.transpose(I)

    # perform stick voting
    '''if verb:
        h = waitbar(0,'Performing voting...')'''
    
    a = 0
    
    for s in I: ##############################################################################################
        a = a + 1
        #print(a)

        #[i, j] = ind2sub(size(l1),s);
        #j = int(s / len(l1))
        #i = s % j - 1
        i = s[0]
        j = s[1]
        #print(i, j)
        # the direction is e1 with intensity l1-l2
        v = e1[i, j, :]   
        Fk = cst.compute_stick_tf(v, N, n, sigma, c)

        Gk = l1[i, j] - l2[i, j]    # the current intensity of the vote
        # select the correct range
        i1 = max(nn - i ,0)
        i2 = max(i + nn - N + 1, 0)
        j1 = max(nn - j, 0)
        j2 = max(j + nn - N + 1, 0)
        #print('i1: ', i1, ' i2: ', i2, ' j1: ', j1, ' j2: ', j2)
        #print('T01: ', int(i-nn+i1), ' T02: ', int(i+nn-i2+1), ' T11: ', int(j-nn+j1), ' T12: ', int(j+nn-j2+1))
        #print('F01: ', int(i1), 'F02: ', int(Fk.shape[0]-i2), 'F11: ', int(j1), ' F12: ', int(Fk.shape[0]-j2))
        
        T[int(i-nn+i1): int(i+nn-i2+1), int(j-nn+j1): int(j+nn-j2+1), :, :] = (T[int(i-nn+i1): int(i+nn-i2+1), int(j-nn+j1): int(j+nn-j2+1), :, :]
                                                                        + Gk * Fk[int(i1): int(Fk.shape[0]-i2), int(j1): int(Fk.shape[0]-j2), :, :])
        #print('------------------------------------')


    return T
