from scipy.stats import multivariate_normal
from sklearn import cluster
import numpy as np
from scipy import linalg

def random_cov(n):
   Q = np.random.randn(n,n)
   eigen_mean = 2 
   diag = np.diagflat(np.abs(eigen_mean + np.random.rand(n,1)))
   A = np.dot(np.dot(Q.T, diag), Q);
   return A 


def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        cv_chol = linalg.cholesky(cv, lower=True)

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def expectation(X, w, means, covars):
    X = np.array(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]

    lpr = (log_multivariate_normal_density(X, means, covars) +
               np.log(w))
   
    logprob = np.log(np.sum(np.exp(lpr), axis=1))   
    llh = np.sum(logprob)
    pp = np.exp(lpr - logprob[:, np.newaxis])
    return pp, llh

def expectation_old(X, w, mean, cov):
    m = np.shape(X)[0]
    n = np.shape(w)[0]
    pp  = np.zeros((m ,n))
    llh = 0

    for k in range(m):
        p = np.zeros(n)

        for i in range(n):
            p[i] = w[i] * multivariate_normal.pdf(X[k], mean=mean[i], cov=cov[i])

        if np.sum(p) == 0:
            print p
        llh = np.log(np.sum(p))
        p = p / np.sum(p)
        pp[k] = p

    return pp, llh


def maximization(X, pp):
    m = np.shape(X)[0]
    d = np.shape(X)[1]
    n_components = np.shape(pp)[1]
    
    n = np.sum(pp, axis=0)
    w = n / m 
    mean = np.dot(pp.T, X) / n[:,np.newaxis] 
    cov = np.zeros((n_components,d,d))

    for i in range(n_components):
        X0 = X - mean[i]
        c = np.sqrt(pp[:, i])
        X0 = np.multiply(X0, c[:,np.newaxis])
        cov[i] = np.dot(np.transpose(X0), X0) / n[i] + np.eye(d) * 1e-3 
    return w, mean, cov


def init(X, n_components):
    mean = cluster.KMeans(
        n_clusters=n_components).fit(X).cluster_centers_

    w  =  np.tile(1.0 / n_components,n_components)

    cv = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    
    if not cv.shape:
        cv.shape = (1, 1)
    cov = np.tile(cv, (n_components, 1,1))

    print 'finish initialzation ...'
    return w, mean, cov

def adaption(X, pp, w, mean, cov):
    m = np.shape(X)[0]
    d = np.shape(X)[1]

    n = np.sum(pp, axis=0)
    e1 = np.dot(pp.T, X) / n[:,np.newaxis]
    e2 = np.array([np.dot(X.T*p, X) for p in pp.T]) / n[:,np.newaxis, np.newaxis]

    tau = 16
    alpha = n / (n + tau)

    
    new_w = alpha * n / m + (1 - alpha) * w
    new_w = new_w / np.sum(new_w)
    new_mean = e1*alpha[:,np.newaxis] + mean *(1-alpha)[:,np.newaxis]
    mu2 = np.array([np.dot(m[:,np.newaxis], m[np.newaxis,:]) for m in mean])
    new_mu2 = np.array([np.dot(m[:,np.newaxis], m[np.newaxis,:]) for m in new_mean])   
    new_cov = e2*alpha[:,np.newaxis, np.newaxis] + (cov + mu2)*(1-alpha)[:, np.newaxis, np.newaxis] - new_mu2

    return new_w, new_mean, new_cov
