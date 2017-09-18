import os
import numpy as np
import matplotlib.pyplot as plt

from week2 import tools

from week2.est_entro import est_entro_JVHW, est_entro_MLE

if os.path.exists('w2_jvhw_estimate_result.txt'):
    os.remove('w2_jvhw_estimate_result.txt')
# Total number of Monte-Carlo trials for each alphabet size
mc_times = 50
# calculate the true differential entropy
true = -1.276263936

# change your sample size
record_n = [10000]


for i in range(0, len(record_n)):
    n = record_n[i]

    # get data
    data = []
    with open('../week1/result/data-{0}.txt'.format(int(n)), 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line))

    # jvhws = []
    # for S in range(0, 400, 1):
    #     samp = np.digitize(data, np.linspace(0, 1, S+1))
    #     JVHW = est_entro_JVHW(samp) - np.log2(S)
    #     jvhws.append(abs(JVHW-true))
    # tools.plotSimplePlot(jvhws, title=str(n), save=True, show=False)
# def hhhh():

    # change your scale and step for S
    for S in range(1, n, 1):

        # load into bins
        dist = np.histogram(data, bins=S, range=(float(0), float(1)))[0]
        dist = [float(dist[j]) for j in range(0, len(dist))]
        dist = np.array(dist)
        dist /= dist.sum()


        # calculate the true differential entropy
        true = -1.276263936
        # true = tools.entropy_true(dist) - np.log2(int(n))
        # print true


        # estimate the differential entropy
        # record_n = np.ceil(S / np.log(S))
        try:
            samp = None
            samp = tools.randsmpl(dist, int(n), mc_times)
        except ValueError:
            print S
            continue
        JVHW = est_entro_JVHW(samp) - np.log2(int(S))
        # print JVHW.mean()

        # calculate the root mean squared errjor
        JVHW_err = tools.getRootMeanSquaredError(JVHW, true)

        with open('w2_jvhw_estimate_result.txt', 'a') as f:
            line = ':'.join([str(n), str(S), str(true), str(tools.getMean(JVHW)), str(JVHW_err)])
            f.write(line)
            f.write('\n')
            f.close()
