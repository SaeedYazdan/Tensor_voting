import numpy as np


class Options:
    def __init__(self):
        self.null = ''
        self.h1 = ''
        self.h2 = ''
        self.h = 0



def compute_grad(M, options=''):

    # compute_grad - compute the gradient of an image using central differences
    #
    # grad = compute_grad(M,options);
    #
    #   'options' is a structure:
    #   - options.h is the sampling step size on both direction (default 1).
    #   - options.h1 is the sampling step size on X direction (default 1).
    #   - options.h2 is the sampling step size on Y direction (default 1).
    #   - options.type is the kind of finite difference.
    #       type==2 is fwd differences, ie.
    #           y(i) = (x(i)-x(i-1))/h, with special
    #           care at boundaries.
    #       type==1 is forward differences bilinearly interpolated in the
    #           middle of each pixel (be aware that you have a shift of 1/2 on X and Y for
    #           the location of the gradient).
    #       type==1 is backward differences bilinearly interpolated in the
    #           middle of each pixel (be aware that you have a shift of -1/2 on X and Y for
    #           the location of the gradient).
    #
    #   Copyright (c) 2004 Gabriel Peyré


    h1 = 0
    
    if options == '':
      options = Options()
      #options.null = 0

    if options.h1 == '':
      options.h1 = 1
        
    h1 = options.h1
    if options.h2 == '':
      options.h2 = 1
    
    h2 = options.h2
    if options.h == '':
      h1 = options.h
      h2 = options.h

    [n, p] = M.shape

    try:
        type0 = options.type
    except:
        type0 = 1

    grad = np.zeros((n, p, 2))

    print('h1', h1)


    ###############################################################
    # new code, use faster 2D differences
    if type0 == 1:
        # central differences on X
        #D1 = np.concatenate((M[1:, :], M[-1, :]), axis=0)
        D1 = np.concatenate((M[1:, :], M[-1, :].reshape((1, -1))), axis=0)
        D2 = np.concatenate((M[0, :].reshape((1, -1)), M[:-1, :]), axis=0)
        grad[:, :, 0] = (D1 - D2) / (2 * h1)
        grad[0, :, 0] = (4 * M[1, :] - 3 * M[0, :] - M[2, :] ) / (2 * h1)
        grad[-1, :, 0] = -(4 * M[-2, :] - 3 * M[-1, :] - M[-3, :] ) / (2 * h1)
        # central differences on Y
        D1 = np.concatenate((M[:, 1:], M[:, -1].reshape((-1, 1))), axis=1)
        D2 = np.concatenate((M[:, 0].reshape((-1, 1)), M[:, :-1]), axis=1)
        grad[:, :, 1] = (D1 - D2) / (2 * h2) 
        grad[:, 0, 1] = (4 * M[:, 1] - 3 * M[:, 0] - M[:, 2] ) / (2 * h2)
        grad[:, -1, 1] = -( 4 * M[:, -2] - 3 * M[:, -1] - M[:, -3] ) / (2 * h2)

        return grad
    
    elif type0 == 2:
        #############################################################################
        # accumulate on Y
        '''MM = (M + [M[:, 2:-1], M[:, -1]]) / 2
        # fwd differences on X
        D1 = [MM[2:-1 ,:]; MM[-1, :]]
        D2 = MM
        grad[:, :, 1] = (D1 - D2) / h1
        #############################################################################
        # accumulate on X
        MM = (M + [M[2:-1, :]; M[-1, :]]) / 2
        # fwd differences on Y
        D1 = [MM[:, 2:-1], MM[:,-1]]
        D2 = MM
        grad[:, :, 2] = (D1 - D2)  /h2'''
        grad = ''
    elif type0 == 3:
        #############################################################################
        # accumulate on Y
        '''MM = (M + [M[:, 1], M[:, 1:-1]]) / 2
        # fwd differences on X
        D1 = MM
        D2 = [MM[1, :]; MM[1:-1, :]]
        grad[:, :, 1] = (D1 - D2) / h
        #############################################################################
        # accumulate on Y
        MM = (M + [M[1, :]; M[1:-1 ,:]]) / 2
        # fwd differences on Y
        D1 = MM
        D2 = [MM[:, 1], MM[:, 1:-1]]
        grad[:, :, 2] = (D1 - D2) / h'''
        grad=''
    else:
        raise Exception('This kind of differences is not supported.')
     

    
    # compute the difference in the center of each square
    h = np.zeros((n, p, 2))
    
    for j in range(p):
        h[:, j, 0] = (grad[:, j, 0] + grad[:, j + 1, 0]) / 2

    for i in range(n):
        h[i, :, 1] = (grad[i, :, 1] + grad[i + 1, :, 1]) / 2 ; kl
    
    if type0 == 2:          # fwd differences
        grad[:n-1, :p-1, :] = h[:n-1, :p-1, :]        
    elif type0 == 3:      # bwd differences
        grad[1:n, 1:p, :] = h[1:n, 1:p, :]   
        
    return grad
