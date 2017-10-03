import numpy as np
import lnn
from week4 import tools
from week4.est_entro import est_entro_JVHW
import logging
import os

# logger configuration
log_file_name = 'w4-comparision.log'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
handler_file = logging.FileHandler(log_file_name)
handler_file.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(handler_file)


def f_smooth(x):
    sigma = 0.0997355701
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


def lnn_smooth(true, filename, mc_times):

    # logger = logging.getLogger()
    logger.info('lnn_smooth begin')
    # change the size of samples
    n_size = [100, 1000, 10000, 100000]

    for n in n_size:
        for mc_time in range(0, mc_times):

            logger.info('lnn_smooth: current n={0}'.format(n))

            # get samples
            data = MCMC_SD(f_smooth, n)
            data = [[data[i]] for i in range(0, len(data))]
            data = np.array(data)
            logger.info('lnn_smooth: get data done')

            # get estimation results
            k = 27
            result = []
            for tr in range(k, 40, 5):
                try:
                    entropy = lnn.LNN_2_entropy(data, k=k, tr=tr, bw=0)
                    # print entropy
                    result.append(entropy)
                except ValueError, e:
                    logger.error('k={0} tr={1} error={2}'.format(k, tr, e))
                except IndexError, e:
                    logger.error('k={0} tr={1} error={2}'.format(k, tr, e))
            logger.info('lnn_smooth: k={0} get entropy done'.format(k))

            # save the results
            result = np.array(result)
            with open(filename, 'a') as f:
                RMSE = tools.getRootMeanSquaredError(result, true)
                f.write(':'.join([str(n), str(k), str(true), str(RMSE)]))
                f.write('\n')
                f.close()
            logger.info('lnn_smooth: write results at {0} done'.format(filename))
            logger.info('lnn_smooth: mc_time={0} done'.format(mc_time))

        logger.info('lnn_smooth: n={0} done'.format(n))

    logger.info('lnn_smooth done')


def jvhw_smooth(true, filename, mc_times):

    # logger = logging.getLogger()
    logger.info('jvhw_smooth begin')

    # change the size of samples
    n_size = [100, 1000, 10000, 100000]

    for n in n_size:

        logger.info('jvhw_smooth: current n={0}'.format(n))

        # get samples
        data = MCMC_SD(f_smooth, n)
        logger.info('jvhw_smooth: get data done')

        if n == 100 or n == 10000 or n == 100000 :
            continue
        else:
            stop = 500

        for S in range(1, stop, 1):
            # load into bins
            dist = np.histogram(data, bins=S, range=(float(0), float(1)))[0]
            dist = [float(dist[j]) for j in range(0, len(dist))]
            dist = np.array(dist)
            dist /= dist.sum()


            # estimate the differential entropy
            try:
                samp = None
                samp = tools.randsmpl(dist, int(n), mc_times)
            except ValueError:
                continue

            JVHW = est_entro_JVHW(samp) - np.log2(int(S))
            logger.info('jvhw_smooth: S={0} get entropy done'.format(S))

            # calculate the root mean squared errjor
            JVHW_err = tools.getRootMeanSquaredError(JVHW, true)

            with open(filename, 'a') as f:
                line = ':'.join([str(n), str(S), str(true), str(tools.getMean(JVHW)), str(JVHW_err)])
                f.write(line)
                f.write('\n')
                f.close()
            logger.info('jvhw_smooth: write results at {0} done'.format(filename))

        logger.info('jvhw_smooth: n={0} done'.format(n))

    logger.info('jvhw_smooth done')


if __name__ == '__main__':
    true = -1.27626388002
    lnn_result_file_name = 'w4_klnn-smooth-estimate-result.txt'
    jvhw_result_file_name = 'w4-jvhw-smooth-estimate-result.txt'

    # delete the files if they exist
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    if os.path.exists(lnn_result_file_name):
        os.remove(lnn_result_file_name)

    if os.path.exists(jvhw_result_file_name):
        os.remove(jvhw_result_file_name)

    jvhw_smooth(true, jvhw_result_file_name, 50)
    # lnn_smooth(true, lnn_result_file_name, 50)