import numpy as np

  
def compute_gaussian_filter(n, s, N):

    # compute_gaussian_filter - compute a 1D or 2D Gaussian filter.
    #
    #   f = compute_gaussian_filter(n,s,N);
    #
    #   'n' is the size of the filter, odd for no phase in the filter.
    #       (if too small it will alterate the filter).
    #       use n=[n1,n2] for a 2D filter
    #   's' is the standard deviation of the filter.
    #   'N' is the size of the big signal/image (supposed to lie in [0,1] or [0,1]x[0,1]).
    #       use N=[N1,N2] for a 2D filter.
    #
    #   The equation (in 1D) is
    #       f[k] = exp( -(x(k)^2/(2*s^2)) );
    #   where x span [-1/2,1/2].
    #
    #   The filter is normalised so that it sums to 1.
    #
    #   Copyright (c) 2004 Gabriel PeyrŽ

    nd = 1
    if len(n) > 1 and n[0] > 1:
        nd = 2

    if nd == 2 and len(s) == 1:
        s = [s[0], s[0]]

    if nd == 2 and len(N) == 1:
        N = [N, N]

    if nd == 1:
        f = build_gaussian_filter_1d(n=n, s=s, N=N)
    else:
        f = build_gaussian_filter_2d(n, s, N)

    return f


def build_gaussian_filter_2d(n, s='', N=''):

    # build_gaussian_filter_2d - compute a 2D Gaussian filter.
    #
    #   f = build_gaussian_filter_2d(n,s,N);
    #
    #   'n' is the size of the filter, odd for no phase in the filter.
    #       (if too small it will alterate the filter).
    #   's' is the standard deviation of the filter.
    #   'N' is the size of the big image (supposed to lie in [0,1]x[0,1]).
    #
    #   The filter is normalised so that it sums to 1.
    #
    #   Copyright (c) 2004 Gabriel PeyrŽ

    if arg_counter(n, s, N) < 2:
        raise Exception('Not enough arguments.')
    
    if arg_counter(n, s, N) < 3:
        N = n

    if len(N) == 1 or N[0] == 1:
        #N = N(:);
        N = [N, N]
        
    if len(s) == 1 or s[0] == 1:
        #s = s(:);
        s = [s, s]
    

    if s[0] <= 0:
        f = np.zeros((n, n))
        f[round((n - 1) / 2), round((n - 1) / 2)] = 1
        return f

    # x = ( (0:n(1)-1)-(n(1)-1)/2 )/(N(1)-1)
    x = [((i - (n[0] - 1) / 2) / (N[0] - 1)) for i in range(n[0])]
    #y = ( (0:n(2)-1)-(n(2)-1)/2 )/(N(2)-1)
    y = [((i - (n[1] - 1) / 2) / (N[1] - 1)) for i in range(n[1])]
    [Y, X] = np.meshgrid(y, x)
    f = np.exp(-(X ** 2 / (2 * s[0] ** 2)) - (Y ** 2 / (2 * s[1] ** 2)))
    f = f / sum(sum(f))
    return f


def build_gaussian_filter_1d(n, s, N):
    # build_gaussian_filter_1d - compute a Gaussian filter.
    #
    #   f = build_gaussian_filter_1d(n,s,N);
    #
    #   Copyright (c) 2004 Gabriel PeyrŽ

    if arg_counter(n, s, N) < 2:
        raise Exception('Not enough arguments.')
    
    if arg_counter(n, s, N) < 3:
        N = n

    if s <= 0:
        f = np.zeros((n,1))
        f[round((n - 1) / 2)] = 1
        return f

    # x = ( (0:n-1)-(n-1)/2 )/(N-1)
    x = [((i - (n - 1) / 2) / (N - 1)) for i in range(n)]
    f = np.exp( -x ** 2 / (2 * s ** 2) )
    f = f / sum(f)
    return f


