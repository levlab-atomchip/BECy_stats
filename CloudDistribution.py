import CloudImage
import numpy as np
import glob
import matplotlib.pyplot as plt


#This is a class definition for getting distributional information over a set of cloud images

class CloudDistributions():
    def __init__(self,dir):
        self.filelist = glob.glob(dir + '*.mat')
        self.numimgs = len(self.filelist)
        self.dists = {}
        self.imagelist = []
        for file in self.filelist:
            self.imagelist.append(CloudImage.CloudImage(file))
        
    def values_metadata(self,var):
        var_dist = []
        for img in self.imagelist:
            try:
                this_value = img.getVariableValues()[var]
            except KeyError:
                print('Invalid Variable Name')
                raise KeyError
            except CloudImage.FitError:
                print('Fit Error')
            var_dist.append(this_value)
        self.dists[var] = var_dist
        
    def values_method(self, var_method, **kwargs):
        var_dist = []
        for img in self.imagelist:
            try:
                this_value = eval('img.' + var_method + '(**kwargs)')
            except AttributeError:
                print('Invalid Method Name')
                raise AttributeError
            var_dist.append(this_value)
        self.dists[var_method] = var_dist
        

    # def savevalues(self,dir,var):
        # outputfile = dir+var+ '.csv'
        # data = self.values(var)
        # with open(outputfile, 'w') as f:
            # writer = csv.writer(f)
            # for num in data[var]:
                # writer.writerow([num])

    def plotdistribution(self,var, **kwargs):
        if self.does_var_exist(var, **kwargs):
            plt.subplot(121)
            plt.hist(self.dists[var],np.ceil(np.power(self.numimgs,0.33)))
            plt.ylabel('Counts')
            plt.xlabel(var)
            plt.title('Histogram')
            plt.subplot(122)
            plt.plot(self.dists[var],marker='o',linestyle='--')
            plt.ylabel(var)
            plt.xlabel('Run Number')
            plt.title('Time Series')
            plt.show()
        else:
            print("Variable Does Not Exist")
        
    def mean(self,var):
        return self.calc_statistic(var, np.mean)
        
    def std(self,var):
        return self.calc_statistic(var, np.std)
    
    def median(self,var):
        return self.calc_statistic(var, np.median)
        
    def calc_statistic(self, var, statistic):
        if self.does_var_exist(var):
            return statistic(self.dists[var])
        else:
            print("Variable Does Not Exist")
            return Null
            
    def does_var_exist(self, var, **kwargs):
        if var in self.dists.keys():
            return True
        else:
            try:
                self.values_metadata(var)
            except KeyError:
                try:
                    self.values_method(var, **kwargs)
                except:
                    print('Invalid Variable')
                    return False
            return True
            
if __name__ == "__main__":
    dir = r'C:\Users\Will\Desktop\Levlab Projects\BECy Data\090313\loadmot_num\2013-09-03\\'
    my_dists = CloudDistributions(dir)
    my_dists.values_method('getAtomNumber', axis=1, offset_switch = True, flucCor_switch = True, debug_flag = False, linear_bias_switch = True)
    my_dists.does_var_exist('getAtomNumber', axis=1, offset_switch = True, flucCor_switch = True, debug_flag = False, linear_bias_switch = True)
    print(my_dists.dists['getAtomNumber'])
    # my_dists.plotdistribution('getAtomNumber', axis=1, offset_switch = True, flucCor_switch = True, debug_flag = False, linear_bias_switch = True)