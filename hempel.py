# module for outlier removal using hempel criterion

import numpy

def hempel_filter(list, nMADM = 3):
    median = numpy.median(list)
    adm = [abs(x - median) for x in list]
    madm = numpy.median(adm)
    cutoff = nMADM*madm
    filtered = []
    filt_ind = []
    for index, item in enumerate(list):
        if abs(item - median) > cutoff:
            filtered.append(item)
            filt_ind.append(index)

    return filtered, filt_ind