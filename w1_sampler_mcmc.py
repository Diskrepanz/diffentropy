"""
sampler implemented with MCMC (Markov Chain Monte Carlo)

author: cyrus
date: 2017-09-06
"""
import json
import time

import numpy as np

import tools


def f(x):
    if 0 <= x < 0.25:
        return float(0)
    elif 0.25 <= x < 0.5:
        return 16.0 * (x - 0.25)
    elif 0.5 <= x < 0.75:
        return -16.0 * (x - 0.75)
    elif 0.75 < x <= 1:
        return float(0)
    else:
        raise ValueError('value should in [0, 1], now is {0}'.format(x))


def MCMC_SD(size=100):
    """
    Assume X~U(0, 1) with only 1 dimension, then generate lots of that X,
    if acceptable, add it to the result set until full.
    """

    result = []

    current = 0.5

    for i in range(0, size):

        next_ = np.random.rand()
        u = np.random.rand()

        if f(next_) == float(0):
            continue
        # print u, min(f(next_) / f(current), 1)

        if u < min(f(next_) / f(current), 1):
            # accept
            result.append(next_)
            current = next_
        else:
            result.append(current)

    return result


if __name__ == '__main__':
    start = time.time()

    size = 100000000
    f_esti = MCMC_SD(size=size)

    tools.plotSimpleKde(f_esti, x_lim=(0, 1), y_lim=(0, 5),
                        title='size={0}'.format(size), save=True, show=False)

    end = time.time()

    with open('running.log', 'a') as f:
        log = {
            'comment': 'fixed bug 5',
            'function': 'sampler',
            'dimension': '1',
            'size': size,
            'running_time': (end - start)
        }
        f.write(json.dumps(log))
        f.write('\n')
        f.flush()
        f.close()
