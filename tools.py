import os

# import pywt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy import signal
from scipy.signal import filter_design as fd
from sklearn import preprocessing as pp


def getShapiroWilk(x):
    return stats.shapiro(x)[1]


def getMean(x):
    array = np.array(x)
    return np.mean(array)


def getStd(x):
    return np.std(x)


def getVar(x):
    array = np.array(x)
    return np.var(array)


def getMedian(x):
    return np.median(x)


def getMode(x):
    t = stats.mode(x)
    return str(t[0][0]) + '(' + str(t[1][0]) + '/' + str(len(x)) + ')'


def getPTP(x):
    return np.ptp(x)


def getZScore(data, mean, std):
    return (data - mean) / std


def getCorrcoef(*lists):
    t = []
    for l in lists:
        t.append(l)
    return np.corrcoef(np.array(t))


def plotHist(data1, data2, title, xlabel, ylabel, legend1, legend2,
             cumulative=False):
    plt.figure()
    # list is ok but needs integer
    d1 = plt.hist(data1, hold=1, label=legend1, alpha=0.8,
                  cumulative=cumulative)
    d2 = plt.hist(data2, label=legend2, alpha=0.5, cumulative=cumulative)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='center right')
    plt.grid(True)
    plt.savefig('hist ' + title)
    # plt.show(block=False)


def plotDatas(n, datas, title=None, xlabel=None, ylabel=None, legend=None):
    plt.figure()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    for i in range(0, n):
        if legend is not None:
            plt.plot([j / 10.0 for j in range(1, 10)], datas[i],
                     label=legend[i])
    plt.legend(loc='center right')
    plt.ylim(0, 1)
    plt.savefig(title)
    plt.close()


def plot(n, x, y, title, xlabel, ylabel, grid=False):
    plt.figure()
    for i in range(0, n):
        plt.plot(x[i], y[i], '^')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(0, 30)
    plt.ylim(0, 1)
    # plt.grid(grid)
    # plt.savefig(title)
    # plt.legend(loc='center right')
    plt.show()


def plotBox(data1, data2, label1, label2, title, xlabel):
    plt.figure()
    plt.boxplot([data1, data2], vert=False, labels=[label1, label2])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig('box ' + title)
    # plt.show(block=False)


def getSortedCumulativeList(a1, a2):
    t = []
    for i in range(0, len(a1)):
        t.append([a1[i], a2[i], ])
    t.sort()

    r = []
    for i in range(0, len(t)):
        sum = 0
        for j in range(0, i + 1):
            sum += t[j][1]
        r.append([t[i][0], sum])

    return r


def getCumulativeList(l):
    r = []
    for i in range(0, len(l)):
        sum = 0
        for j in range(0, i + 1):
            sum += l[j]
        r.append(sum)

    return r


def getIncrementList(l):
    global sum
    r = []
    for i in range(0, len(l)):
        if i != 0:
            sum = l[i] - l[i - 1]
            r.append(sum)

    return r


def plotFFT(n, l, title, ret=False):
    plt.figure()
    for i in range(0, n):
        length = len(l[i])
        if 128 < length <= 256:
            # to 256 point
            for j in range(0, 256 - length):
                l[i].append(0)
        fft = np.fft.fft(l[i])
        a = [abs(fft[i]) for i in range(0, len(fft))]
        plt.plot(np.linspace(0, 2, len(fft)), a)
    # plt.ylim(0, 50)
    plt.xlim(0, 1)
    plt.title(title)
    plt.savefig(
        os.path.join('/home/cyrus/PycharmProjects/data_acc_r/fft/', title))
    plt.close('all')
    # plt.show(block=False)


def getFFT(l):
    length = len(l)
    if length <= 256:
        # to 256 point
        for i in range(0, 256 - length):
            l.append(0)
    fft = np.fft.fft(l)
    a = [abs(fft[i]) for i in range(0, len(fft))]
    return a


def getHPF(n, datas, ftype='cheby2'):
    # iir filter parameter
    Wp = 0.1
    Ws = 0.01
    Rp = 1
    As = 3
    # filters ellip cheby2 cheby1 butter bessel
    b, a = fd.iirdesign(Wp, Ws, Rp, As, ftype=ftype)
    r = []
    for i in range(0, n):
        t = signal.filtfilt(b, a, datas[i])
        r.append(t.tolist())
        # fir filter parameter
        # f = 0.2 # cutoff = f * nyq
        # for i in range(0, n):
        #     b = signal.firwin(3, f, pass_zero=False, window='hamming')
        #     b = signal.firwin(7, f, pass_zero=False, window='blackman')
        #     b = signal.firwin(9, f, pass_zero=False, window='hann')
        # t = signal.filtfilt(b, [1], datas[i])
        # r.append(t.tolist())
    return r


def plotHist2d(x, y):
    """

    """
    plt.hist2d(x, y, bins=30)
    plt.show()


def plotKde(x, title, **kwargs):
    plt.figure()
    sns.set_palette('hls')
    plt.title(title)

    # this is a type of usage of **kwargs
    bins = kwargs.pop('bins', None)
    legend = kwargs.pop('legend', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    loc = kwargs.pop('loc', None)
    xlim = kwargs.pop('xlim', None)
    save = kwargs.pop('save', False)
    show = kwargs.pop('show', True)
    block = kwargs.pop('block', True)
    text = kwargs.pop('text', None)
    xlog = kwargs.pop('xlog', False)

    if legend is not None:
        sns.distplot(x, bins=bins, hist_kws={'label': legend},
                     kde_kws={'label': legend})
    else:
        sns.distplot(x, bins=bins)
        # plt.hist(x)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if loc is not None:
        plt.legend(loc=loc)
    if xlim is not None:
        plt.xlim(xlim)
    if text is not None:
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.text(xmax * 0.01, ymax * 0.6, text)
    if xlog:
        plt.xscale('log')
    if save:
        plt.savefig(' '.join(['kde', title]))
    if show:
        plt.show(block=block)
    plt.close()
    return None


def plotSimpleKde(x, title=None, x_label=None, y_label=None, save=False,
                  x_lim=None, y_lim=None, show=True):
    plt.figure()
    sns.set_palette('hls')
    sns.distplot(x, bins=100, hist_kws={'color': 'y'},
                 kde_kws={'color': 'y'})

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if x_lim is not None:
        plt.xlim(x_lim)

    if y_lim is not None:
        plt.ylim(y_lim)

    if save:
        if title is None:
            raise ValueError('Title is not set')
        plt.savefig(title)

    if show:
        plt.show()

    plt.close()


def plotSimpleBox(x):
    plt.boxplot(x)
    plt.show()


def getPercentile(x, percentile):
    return np.percentile(x, percentile)


def getNormalized(x):
    return pp.normalize(x)


def plotSimplePlot(x,
                   y=None,
                   title=None,
                   xlegend=None,
                   ylegend=None,
                   xlabel=None,
                   ylabel=None):
    plt.plot(x, 'r*', label=xlegend)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if y is not None:
        plt.plot(y, 'bo', label=ylegend)
    plt.legend(loc='center right')
    plt.show()


def getSorted(l):
    length = len(l)
    r = []
    for i in range(0, length):
        min_ = min(l)
        r.append(min_)
        l.remove(min_)
    return r


def plotSimpleXY(x, y, title,
                 x1=None, y1=None, xlabel=None, ylabel=None,
                 legend1=None, legend2=None,
                 show=True, save=False):
    plt.title(title)
    plt.plot(x, y, 'o', label=legend1)
    if x1 is not None and y1 is not None:
        plt.plot(x1, y1, '^', label=legend2)
    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.legend(loc='upper center')
    if save:
        plt.savefig(title)
    if show:
        plt.show()


def plotHeat(x, title):
    cmap = sns.cubehelix_palette(n_colors=30, start=3, rot=5, dark=0.6,
                                 reverse=True,
                                 gamma=2, as_cmap=True)
    mask = None
    # mask = np.ones_like(x)
    # mask[getCrossZeroIndices(x)] = False
    if type(x) is list:
        x = np.array(x)
        x = x.T
    sns.heatmap(x, linewidths=0.05, cmap=cmap, center=0, annot=True, mask=mask)
    plt.yticks([float(i) + 0.5 for i in range(0, 11)],
               [str(i / 10.0) for i in range(0, 11)],
               rotation=0)
    plt.xticks([float(i) + 0.5 for i in range(0, 11)],
               [str(i / 10.0) for i in range(0, 11)],
               rotation=0)
    plt.xlabel('threshold')
    plt.ylabel('acceptance')
    plt.title(title)
    plt.show()
    # plt.savefig(title)


def getCrossZeroIndices(table):
    """
    :type table: list
    """
    array_row = []
    array_col = []

    before = 1
    after = 1
    for row in table:
        for item in row:
            if item * before > 0:
                after *= 1
            else:
                after *= -1
            if 0 > after * before:
                array_row.append(table.index(row))
                array_col.append(row.index(item))
            before = after
    return np.array(array_row), np.array(array_col)
