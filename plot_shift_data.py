import cloud_distribution as CD
import image_align as ia
import psf
import matplotlib.pyplot as plt
import numpy as np

#data1208=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/2014-12-08/')
data0_3=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/0um3/')
'''data0_1=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/0um1/')
data0_2=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/0um2/')
data3_1=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/3um1/')
data3_2=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/3um2/')
data15_1=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/1.5um1/')
data15_2=CD.CD(r'/Users/shenglanqiao/Documents/LevLab/ACM/test_data/resolution/1.5um2/')'''

shifts=[]
_,shifts0_3 = psf.align_lds(data0_3)
shifts.extend(shifts0_3)
'''_,shifts0_1=psf.align_lds(data0_1)
shifts.extend(shifts0_1)
_,shifts0_2=psf.align_lds(data0_2)
shifts.extend(shifts0_2)
_,shifts3_1=psf.align_lds(data3_1)
shifts.extend(shifts3_1)
_,shifts3_2=psf.align_lds(data3_2)
shifts.extend(shifts3_2)
_,shifts15_1=psf.align_lds(data15_1)
shifts.extend(shifts15_1)
_,shifts15_2=psf.align_lds(data15_2)
shifts.extend(shifts15_2)'''

'''psf.sh_stats(data1208)
psf.sh_stats(data1216)
psf.sh_stats(data15_2)'''


plt.figure()
psf.plt_lds(data0_3)
#plt.figure()
#psf.plt_lds(data1216)

#plt.figure()

#psf.plt_lds(data0_1)

print np.std(np.array(shifts))
print np.mean(np.array(shifts))

plt.figure()
plt.ylabel('Shifts (pixels)')
plt.xlabel('Run number')
plt.plot(shifts)

plt.figure()
plt.hist(np.array(shifts), 10)
plt.xlim(-20,20)
plt.xlabel('Shifts (pixels)')
plt.ylabel('Counts')
plt.title('Histogram of shifts over %d runs'%len(shifts))


#plt.plot(shifts)
plt.show()