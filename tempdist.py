import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
import fittemp
import os.path
import re



# User should change this path
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\statistics\\tempdata 082713\\subdop_temp\\2013-08-23\\'

imagelist = glob.glob(dir + '*.mat')

# Regular Expression for extracting timestamps
FILE_RE = re.compile(r'.(\d{6})\.mat')

#initialize
widthsX = []
widthsY = []
TX = []
chi2X = []
TY = []
chi2Y = []
TOFs = []
timestamps = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisfilename = os.path.basename(img)
    thistimestamp = FILE_RE.search(thisfilename).group(1) #so crude!
    thisimg = CloudImage.CloudImage(img)
    try:
        thiscoefs = thisimg.fitGaussian1D(sum(thisimg.getODImage(),0))
    except:
        continue
    thiswidthX = thiscoefs[2]
    try:
        thiscoefs = thisimg.fitGaussian1D(sum(thisimg.getODImage(),1))
    except:
        continue
    thiswidthY = thiscoefs[2]
    thisTOF = thisimg.CurrTOF
    # print(thisTOF)
    TOFs.append(thisTOF)
    timestamps.append(thistimestamp)
    widthsX.append(thiswidthX)
    widthsY.append(thiswidthY)
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

imagewdata = zip(timestamps, TOFs, widthsX, widthsY)
imagewdata.sort(key=lambda tup: tup[0]) #sort by timestamp

tempseq = []
seqs = []
last_TOF = -1
for img in imagewdata:
    this_TOF = img[1]
    if this_TOF < last_TOF:
        seqs.append(tempseq)
        tempseq = []
        tempseq.append(img)
    else:
        tempseq.append(img)
    last_TOF = this_TOF
seqs.append(tempseq)

for seq in seqs:
    # print(seq)
    thisTOFs = [img[1] for img in seq]
    print thisTOFs
    thiswidthsX = [img[2] for img in seq]
    thiswidthsY = [img[3] for img in seq]
    thischi2X, thisTX = fittemp.temp_chi_2(thisTOFs, thiswidthsX)
    thischi2Y, thisTY = fittemp.temp_chi_2(thisTOFs, thiswidthsY)
    TX.append(thisTX)
    TY.append(thisTY)
    chi2X.append(thischi2X)
    chi2Y.append(thischi2Y)
    
# outlier removal
# temps = hempel.hempel_filter(temps)
# print temps
    
outputfile = dir + 'temps' + '.csv'
outputdata = zip(TX, TY, chi2X, chi2Y)
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    for row in outputdata:
        writer.writerow(row)
    
    
# from scipy.stats import gaussian_kde
# density = gaussian_kde(temps)
# xs = np.linspace(.75*np.min(temps),1.25*np.max(temps),200)
# density.covariance_factor = lambda : .25
# density._compute_covariance()
# plt.plot(xs,density(xs))
# plt.xlabel('Cloud Temperature')
# plt.ylabel('Probability Density')
# plt.title('Temperature Probability Density, 100 Runs')
# plt.show()

temps = TX
# outlier removal
temps = hempel.hempel_filter(temps)
# print temps
print(temps)
print(chi2X)
print('TX stats')
print(np.mean(temps))
print(np.std(temps))
print(2*np.std(temps)/np.mean(temps))
plt.hist(TX,20)
plt.title('X Temperature Probability Density')
plt.xlabel('Cloud Temperature')
plt.ylabel('Probability Density')
plt.show()
plt.hist(chi2X, 20)
plt.title('X Temperature Goodness of Fit Probability Density')
plt.xlabel('Goodness of Fit')
plt.ylabel('Probability Density')
plt.show()

temps = TY
# outlier removal
temps = hempel.hempel_filter(temps)
# print temps
print('TY stats')
print(np.mean(temps))
print(np.std(temps))
print(2*np.std(temps)/np.mean(temps))
plt.hist(TY,20)
plt.title('Y Temperature Probability Density')
plt.xlabel('Cloud Temperature')
plt.ylabel('Probability Density')
plt.show()
plt.hist(chi2Y, 20)
plt.title('Y Temperature Goodness of Fit Probability Density')
plt.xlabel('Goodness of Fit')
plt.ylabel('Probability Density')
plt.show()



# plt.plot(temps, marker='o', linestyle = '--')
# plt.show()