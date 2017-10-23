from numpy import exp, log, where, dot, sum, hstack

logistic = (lambda x: 1/(1+exp(-x)), lambda x,f: f*(1-f))
relu = (lambda x: where(x>0, x, 0), lambda x,f: where(x>0, 1., 0.))
ident = (lambda x: x, lambda x,f: 1)
squareloss = lambda target: (lambda x: sum((x-target)**2), lambda x,f: 2*(x-target))

def fwprop(xin, W, f):
    z  = []
    X = [ xin ]
    for (Wk, (fk, _)) in zip(W, f):
        X[-1] = hstack([X[-1], [1]])
        z.append(dot(Wk.T, X[-1]))
        X.append(fk(z[-1]))
    return X, z

def backprop(X, W, f, z, loss):
    lossf, dloss = loss
    Y = X[-1]
    lossY = lossf(Y)
    dlossY = dloss(Y, lossY)
    g = []
    dloss_dX = dlossY
    for (Xk, Wk, (fk, dfk), zk) in zip(X, W, f, z)[::-1]:
        dloss_dz = dloss_dX[:len(zk)] * dfk(zk, fk(zk))
        dloss_dW = dot(transpose([Xk]), [dloss_dz])
        g.append(dloss_dW)
        dloss_dX = dot(Wk, dloss_dz)
    return g[::-1]

def sgd(W, f, loss, alpha, xtrain, ytrain):
    X, z = fwprop(xtrain, W, f)
    g = backprop(X, W, f, z, loss)
    W = [ Wk - alpha * gk for (Wk, gk) in zip(W, g) ]
    return W, g

def randomweights(dims):
    W = [ randn(rows + 1, cols) for (rows, cols) in zip(dims[:-1], dims[1:]) ]

