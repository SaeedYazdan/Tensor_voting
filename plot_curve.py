def plot_curve(plt, c='', M='', st=''):

    # plot_curve - plot a 2d curve.
    # 
    # plot_curve(c, M,st);
    #
    #   Note that 'c' can also be a cell array of curves.
    #   - 'M' is a [n,p] image put in background (ranging in [0,1]x[0,1])
    #   - 'c' is of size 2xm (ranging in [0,1]x[0,1])
    #   - 'str' is the style of the curve.
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if st == '':
        st = 'red'
    if M == '':
        M = []

    if M != '':
        [n, p] = M.shape
        #x = 0:1 / (n - 1):1
        x = [i/(n-1) for i in range(n)]
        #y = 0:1 / (n - 1):1 # ( 0:1/(p-1):1 )* (p-1)/(n-1);
        y = [i/(n-1) for i in range(n)]
        # display image in X/Y frame
        M = M.T
        # M = M(end:-1:1,:);
        plt.imshow(M)
        #axis image

    if c != '':
        for line in c:
            plt.plot(line.T[0, :]*(n), line.T[1,:]*(p), st)
        #pass
    else:
        if st != '':
            str1 = st;
            del st
            for i in range(len(c)):
                st[i] = str1        
        
        for i in range(len(c)):
            #plt.plot( c[i][1,:], c[i][2, :], st[i] )
            pass
    
    #axis tight
    #hold off
