import numpy as np

import tools


def f(x):
    if 0 <= x < 0.25:
        return float(0)
    elif 0.25 <= x < 0.5:
        return 16.0 * (x - 0.25)
    elif 0.5 <= x < 0.75:
        return -16.0 * (x - 0.75)
    elif 0.75 <= x <= 1:
        return float(0)
    else:
        raise ValueError('value should in [0, 1], now is {0}'.format(x))


def f_smooth(x):
    sigma = 0.0988
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * np.power(x - 0.5, 2) / np.power(sigma, 2))


def MCMC_SD(f, size=100):
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
    # plot pdf
    # x = np.linspace(0, 1, 101)
    # y_smooth = [f_smooth(x[i]) for i in range(0, len(x))]
    # y_original = [f(x[i]) for i in range(0, len(x))]
    #
    # plt.plot(x, y_smooth)
    # plt.plot(x, y_original)
    # plt.legend(['original', 'smooth'], loc='center right')
    # title = 'probability density function comparision'
    # plt.title(title)
    # plt.show()
    # plt.savefig(title)

    # get samples and kde
    # for n in (100, 1000, 10000, 100000):
    #     samples = MCMC_SD(f_smooth, n)
    #     tools.plotSimpleKde(samples, title='f_smooth kde size={0}'.format(n), save=True, show=False)
    pass


if __name__ == '__main__':
    main()
