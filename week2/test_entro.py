# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from Python import tools
from est_entro import est_entro_JVHW, est_entro_MLE


def entropy_true(p):
    """computes Shannon entropy H(p) in bits for the input discrete distribution.

    This function returns a scalar entropy when the input distribution p is a
    vector of probability masses, or returns in a row vector the columnwise
    entropies of the input probability matrix p.
    """
    validate_dist(p)
    # t1 =  -np.log2(p ** p)
    # t2 = -np.log2(p[0] ** p[0])
    # t = t1.sum(axis=0)
    return -np.log2(p ** p).sum(axis=0)


def validate_dist(p):
    if np.imag(p).any() or np.isinf(p).any() or np.isnan(p).any() or (
                p < 0).any() or (p > 1).any():
        raise ValueError(
            'The probability elements must be real numbers between 0 and 1.')

    eps = np.finfo(np.double).eps
    if (np.abs(p.sum(axis=0) - 1) > np.sqrt(eps)).any():
        raise ValueError('Sum of the probability elements must equal 1.')


def randsmpl(p, m, n):
    validate_dist(p)

    # p.cumsum() => return cumulative sum
    # r_ => add a 0 in front of p.cumsum()
    edges = np.r_[0, p.cumsum()]
    # machine limit for floating point-types
    eps = np.finfo(np.double).eps
    if np.abs(edges[-1] - 1) > np.sqrt(eps):
        edges = edges / edges[-1]
    edges[-1] = 1 + eps
    # tools.plotSimpleKde(edges, x_lim=(0, 1), y_lim=(0, 5),
    #                     title='edges S = {0}'.format(int(len(p))), save=True,
    #                         show=False)
    # tools.plotSimplePlot(edges, save=True, title='edges y=(x)', show=False)

    return np.digitize(np.random.rand(m, n), edges)


if __name__ == '__main__':
    C = 1
    num = 15
    mc_times = 50  # Total number of Monte-Carlo trials for each alphabet size
    # y = ceil(x) => min(integer y) >= x
    # logspace(2, 6, num)  => G.P. from 10^2 to 10^6 s.t. N=num
    # log(x) => ln(x)
    record_S = np.ceil(np.logspace(2, 6, num))
    record_n = np.ceil(C * record_S / np.log(record_S))

    true_S = np.zeros(num)
    JVHW_err = np.zeros(num)
    MLE_err = np.zeros(num)

    twonum = np.random.rand(2, 1)
    for i in range(num):
        S = record_S[i]
        n = record_n[i]
        print("S = {0}, n = {1}".format(int(S), int(n)))

        # sample one
        # sample from beta distribution with alpha=twonum[0] beta=twonum[1] N=S
        dist = np.random.beta(twonum[0], twonum[1], int(S))
        # tools.plotSimpleKde(dist, x_lim=(0, 1), y_lim=(0, 5),
        #                     title='S = {0} n {1}'.format(int(S), int(n)), save=True,
        #                         show=False)
        # normalize
        dist /= dist.sum()

        true_S[i] = entropy_true(dist)
        samp = randsmpl(dist, int(n), mc_times)
        # tools.plotSimpleKde(samp[0], x_lim=(0, 1), y_lim=(0, 5),
        #                     title='samp[0] S = {0}'.format(int(S)), save=True,
        #                         show=False)
        # tools.plotSimplePlot(samp[0], save=True, title='samp y=(x)')

        record_JVHW = est_entro_JVHW(samp)
        record_MLE = est_entro_MLE(samp)

        JVHW_err[i] = np.mean(np.abs(record_JVHW - true_S[i]))
        MLE_err[i] = np.mean(np.abs(record_MLE - true_S[i]))

    # plt.plot(record_S / record_n, JVHW_err, 'b-s', linewidth=2,
    #          markerfacecolor='b')
    # plt.plot(record_S / record_n, MLE_err, 'r-.o', linewidth=2,
    #          markerfacecolor='r')
    # plt.legend(['JVHW', 'MLE'], loc='upper left')
    # plt.xlabel('S/n')
    # plt.ylabel('Mean Absolute Error')
    # plt.title('Entropy Estimation')
    # plt.xlim(4, 14.5)
    # plt.show()
