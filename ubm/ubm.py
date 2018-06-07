import sys
import numpy as np
import time
from core import expectation, maximization, init, adaption
from audio.timit_reader import read_timit
from audio.signals import mfcc
import yaml
import glob

def train_ubm(X, n_components, epslion=1e-6, max_iters = 500, save_step=1, save_path=None, save_name=None):
    m, d = np.shape(X)

    w, mean, cov = init(X, n_components)
    pre_llh = sys.float_info.min 
    n_iter = 0
    
    print "instance size {}, demensions {}, number of clusters {}".format(m, d, n_components)

    start = time.time()
    pp, llh = expectation(X, w, mean, cov) 
    llh = np.sum(llh)

    while n_iter < max_iters and np.abs(llh - pre_llh) >= epslion:
        pre_llh = llh
        
        middle = time.time()
        w, mean, cov = maximization(X, pp)
        end = time.time()
        print "iteratrion {}:".format(n_iter)
        print "log likehood: {}, expectation take: {}s, maximization take: {}s".format(llh, middle - start, end - middle)
        start = time.time()
        pp, llh = expectation(X, w, mean, cov)
        llh = np.sum(llh)

        n_iter = n_iter + 1

        if save_step >= 1 and n_iter % save_step == 0 and save_path != None:
            model = {'w':w, 'mean':mean, 'cov':cov}
            full_path = os.join(save_path, save_name, '_' ,n_iter, '.yml')
            with open(save_path, 'wb') as f:
                yaml.dump(model, f, default_flow_style=False)
              
    return w, mean, cov

def adapt_ubm(X, weight, means, covars, epslion=1e-6, max_iters=500):
    pre_llh = sys.float_info.min 
    pp, llh = expectation(X, weight, means, covars)
    llh = np.sum(llh)
    print llh
    n_iter = 0 

    while n_iter < max_iters and np.abs(llh - pre_llh) >= epslion:
        pre_llh = llh
        weight, means, covars = adaption(X, pp, weight, means, covars)
        pp, llh = expectation(X, weight, means, covars)
        llh = np.sum(llh)
        print llh

    return weight, means, covars

def score(X, weight, means, covars,):
    pass

def preprocess(path):
    wav_path = glob.glob(path)
   
    feat = np.empty([0, 20]) 
    for p in wav_path:
       sr, data = read_timit(wav_path[0])
       mfcc_feat = mfcc(data, sr, int(sr*0.025), int(sr*0.01))
       feat = np.concatenate((feat, mfcc_feat.T))
       print p
    return feat   

def load_ubm(path):
    with open(path, 'rb') as f:
        model = yaml.load(f)
        weight = model['w']
        means = model['mean']
        covars = model['cov']
        return weight, means, covars 
