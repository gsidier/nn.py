from numpy import exp, log, where, dot, sum, hstack, abs, transpose
from numpy.random import randn

logistic = (lambda x: 1/(1+exp(-x)), lambda x,f: f*(1-f))
relu = (lambda x: where(x>0, x, 0), lambda x,f: where(x>0, 1., 0.))
ident = (lambda x: x, lambda x,f: 1)
squareloss = lambda target: (lambda x: sum((x-target)**2), lambda x,f=None: 2*(x-target))
logloss = lambda target: (lambda y: sum(-target*log(y) - (1-target)*log(1-y)), lambda y,f=None: -target/y + (1-target)/(1-y))

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
    return [ randn(rows + 1, cols) for (rows, cols) in zip(dims[:-1], dims[1:]) ]

if __name__ == '__main__':

# test gradient calc

	eps = 1e-2
	
	dims = [5, 20, 2]
	W = randomweights(dims)
	f = [ logistic, logistic, logistic ]
	
	xin = randn(dims[0])
	X, z = fwprop(xin, W, f)

	target = squareloss(randn(dims[-1]))
	g = backprop(X, W, f, z, target)
	
	for k, (Wk, gk) in enumerate(zip(W, g)):
		for i in xrange(Wk.shape[0]):
			for j in xrange(Wk.shape[1]):
				Wup = [ _ for _ in W ]
				Wup[k] = Wk.copy()
				Wup[k][i, j] += eps
				Xup, _ = fwprop(xin, Wup, f)
				Wdn = [ _ for _ in W ]
				Wdn[k] = Wk.copy()
				Wdn[k][i, j] -= eps
				Xdn, _ = fwprop(xin, Wdn, f)
				dloss_dWij = (target[0](Xup[-1]) - target[0](Xdn[-1])) / (2*eps)
				assert(abs(dloss_dWij - gk[i, j]) < 1e-4)
				
	print "All tests passed."
