import numpy as np
from week3 import lnn, tools
import os


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

    return result


def main():
    # change the size of samples
    size = 100000
    data = MCMC_SD(size)
    data = [[data[i]] for i in range(size)]
    data = np.array(data)
    true = -1.276263936

    # for k in (24):
    k = 27
    result = []
    for tr in range(k, 40, 5):
        try:
            entropy = lnn.LNN_2_entropy(data, k=k, tr=tr, bw=0)
            print entropy
            result.append(entropy)
            # print 'k={0} tr={1} error={2}'.format(k, tr, error)
        except ValueError, e:
            print 'k={0} tr={1} error={2}'.format(k, tr, e)
        except IndexError, e:
            print 'k={0} tr={1} error={2}'.format(k, tr, e)

    result = np.array(result)
    with open('w3_klnn-estimate-result', 'a') as f:
        print result-true
        RMSE = tools.getRootMeanSquaredError(result, true)
        f.write(':'.join([str(size), str(k), str(RMSE)]))
        f.write('\n')
        f.close()
    print 'write for k={0} done'.format(k)
    return tools.getRootMeanSquaredError(result, true)


if __name__ == '__main__':
    # if os.path.exists('w3_klnn-estimate-result'):
    #     os.remove('w3_klnn-estimate-result')
    results = []
    # repeat for 50 times
    for i in range(0, 50):
        results.append(main())
    print tools.getMean(results)
