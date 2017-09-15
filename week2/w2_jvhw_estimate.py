import numpy as np
import matplotlib.pyplot as plt

from week2 import tools
from week2.test_entro import entropy_true, randsmpl
from week2.est_entro import est_entro_JVHW, est_entro_MLE

C = 1
num = 4
mc_times = 50  # Total number of Monte-Carlo trials for each alphabet size

record_S = np.array([100, 1000, 10000, 100000])
record_n = np.ceil(C * record_S / np.log(record_S))

true_S = np.zeros(num)
JVHW_err = np.zeros(num)
MLE_err = np.zeros(num)

for i in range(num):
    S = record_S[i]
    n = record_n[i]
    print("S = {0}, n = {1}".format(int(S), int(n)))

    dist = []
    with open('../week1/result/data-{0}.txt'.format(int(S)), 'r') as f:
        lines = f.readlines()
        for line in lines:
            dist.append(float(line))

    # normalize
    dist = np.array(dist)
    dist /= dist.sum()

    true_S[i] = entropy_true(dist)
    print 'TRUE ENTROPY {0}'.format(true_S[i])
    samp = randsmpl(dist, int(n), mc_times)

    record_JVHW = est_entro_JVHW(samp)
    record_MLE = est_entro_MLE(samp)
    print 'JVHW {0}'.format(tools.getMean(record_JVHW))
    print 'MLE {0}'.format(tools.getMean(record_MLE))

    # JVHW_err[i] = np.mean(np.abs(record_JVHW - true_S[i]))
    # MLE_err[i] = np.mean(np.abs(record_MLE - true_S[i]))

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