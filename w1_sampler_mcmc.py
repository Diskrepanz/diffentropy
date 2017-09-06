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
    if acceptable, add it to the result set, if not, add last X.
    """

    result = []

    current = 0.5

    reject = 0

    for i in range(0, size):

        next_ = np.random.rand()
        u = np.random.rand()

        if f(current) == float(0):
            condition = 0
        else:
            condition = min(f(next_) / f(current), 1)

        if u < condition:
            # accept
            result.append(next_)
            current = next_
        else:
            # refuse
            result.append(current)
            reject += 1

    return result, reject / float(size)


if __name__ == '__main__':
    start = time.time()

    # settings
    size = 100
    save_data = True
    plot = False
    log = False

    f_esti, reject_rate = MCMC_SD(size=size)

    if plot:
        tools.plotSimpleKde(f_esti, x_lim=(0, 1), y_lim=(0, 5),
                            title='size={0}'.format(size), save=True, show=False)

    end = time.time()

    # save data
    if save_data:
        with open('data-{0}.txt'.format(size), 'w') as f:
            data = '\n'.join([str(f_esti[i]) for i in range(0, len(f_esti))])
            f.write(data)
            f.close()

    # log
    if log:
        with open('running.log', 'a') as f:
            log = {
                'comment': 'final-sampler',
                'function': 'sampler',
                'dimension': '1',
                'size': size,
                'reject_rate': reject_rate,
                'running_time': (end - start)
            }
            f.write(json.dumps(log))
            f.write('\n')
            f.flush()
            f.close()
