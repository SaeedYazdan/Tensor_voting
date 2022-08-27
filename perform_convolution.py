import numpy as np
from scipy import signal



def perform_convolution(x, h, bound = 'sym'):

    # perform_convolution - compute convolution with centered filter.
    #
    #   y = perform_convolution(x,h,bound);
    #
    #   The filter 'h' is centred at 0 for odd
    #   length of the filter, and at 1/2 otherwise.
    #
    #   This works either for 1D or 2D convolution.
    #   For 2D the matrix have to be square.
    #
    #   'bound' is either 'per' (periodic extension) 
    #   or 'sym' (symmetric extension).
    #
    #   Copyright (c) 2004 Gabriel Peyré

    n = x.shape[0]
    p = h.shape[0]

    y = 0

    bound = bound.lower()

    nd = len(x.shape)
    if x.shape[0]==1 or x.shape[1]==1:
        nd = 1

    if bound == 'sym':
        
        ##################################################
        # symmetric boundary conditions
        
        d1 = int(p / 2)           # padding before
        d2 = p - d1 - 1           # padding after
        
        if nd == 1:
            ############################ 1D ##############################
            # x = x(:)
            # h = h(:)
            # xx = [ x[d1:-1:1]; x; x[end:-1:end-d2+1] ]
            x1 = x[:d1][::-1]
            x2 = x[-d2:][::-1]
            x3 = np.concatenate((x1, x), axis=0)
            xx = np.concatenate((x3, x2), axis=0)
            y = signal.convolve2d(xx, h)
            y = y[p:-p + 1]
        elif nd == 2:
            ############################ 2D ##############################
            # double symmetry
            xx = x
            #xx = [ xx[d1:-1:1,:]; xx; xx[end:-1:end-d2+1,:] ]
            x1 = xx[:d1, :][::-1]
            x2 = xx[-d2:, :][::-1]
            x3 = np.concatenate((x1, xx), axis=0)
            
            xx = np.concatenate((x3, x2), axis=0)
            
            #xx = [ xx[:,d1:-1:1], xx, xx[:,end:-1:end-d2+1] ]
            x1 = xx[:, :d1]#[::-1]
            x1 = np.fliplr(x1)
            x2 = xx[:, -d2:]#[::-1]
            x2 = np.fliplr(x2)
            x3 = np.concatenate((x1, xx), axis=1)
            xx = np.concatenate((x3, x2), axis=1)
            
            #y = np.conv2(xx, h)
            y = signal.convolve2d(xx, h)
            y = y[(2 * d1):(2 * d1 + n), (2 * d1):(2 * d1 + n)]

    else:
        ##################################################
        # periodic boundary conditions
        
        if p > n:
            raise Exception('h filter should be shorter than x.')
        
        d = int((p - 1) / 2)
        '''if nd == 1: 
            # x = x(:)
            # h = h(:)
            h = [ h[d+1:-1]; zeros(n-p,1); h[1:d] ]
            y = real( ifft( fft(x).*fft(h) ) )
        else:
            h = [ h[d+1:-1,:]; zeros(n-p,p); h[1:d,:] ]
            h = [ h[:,d+1:-1], zeros(n,n-p), h[:,1:d] ]
            y = real( ifft2( fft2(x).*fft2(h) ) )'''
        y=0
        
    return y
