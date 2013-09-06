import CloudImage
import numpy as np

#This is a class definition for getting distributional information over a set of cloud images

class CloudDistributions():
    def __init__(self,dir):
        imagelist = glob.glob(dir + '*.mat')
        numimgs = len(imagelist)
        
    def values(self,var):
        data = {}
        for img in imagelist:
            thisimg = CloudImage.CloudImage(img)
            thisvalue = thisimg.getvalue(var)
            np.append(data[var], [thisvalue])
            #data[var].append(thisvalue)
        return data

    def savevalues(self,dir,var):
        outputfile = dir+var+ '.csv'
        data = self.values(var)
        with open(outputfile, 'w') as f:
            writer = csv.writer(f)
            for num in data[var]:
                writer.writerow([num])

    def plotdistribution(self,var):
        data = self.values(var)    
        plt.subplot(121)
        plt.hist(data[var],np.ceil(np.power(numimgs,0.33)))
        plt.ylabel('Counts')
        plt.xlabel(var)
        plt.title('Histogram')
        plt.subplot(122)
        plt.plot(data[var],marker='o',linestyle='--')
        plt.ylabel(var)
        plt.xlabel('Run Number')
        plt.title('Time Series')
        plt.show()
        
    def mean(self,var):
        data = self.values(var)
        return np.mean(data[var])
        
    def std(self,var):
        data = self.values(var)
        return np.var(data[var])
    
    def median(self,var):
        return np.median(self.values[var])
        