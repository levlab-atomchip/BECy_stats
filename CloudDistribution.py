import CloudImage
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import stats
import math

import win32gui
from win32com.shell import shell, shellcon



#This is a class definition for getting distributional information over a set of cloud images
# Right now this can provide the functionality of:
# numberhist, posdist, widthdist, intensitydist, numpos, paramregress, pos_regress, posxvy, twoparamregress

class CloudDistributions():
    def __init__(self):
    
        desktop_pidl = shell.SHGetFolderLocation (0, shellcon.CSIDL_DESKTOP, 0, 0)
        pidl, display_name, image_list = shell.SHBrowseForFolder (
          win32gui.GetDesktopWindow (),
          desktop_pidl,
          "Choose a folder",
          0,
          None,
          None
        )
        self.dir = shell.SHGetPathFromIDList (pidl)
        print(self.dir)
    
        self.filelist = glob.glob(self.dir + '\\*.mat')
        self.numimgs = len(self.filelist)
        self.dists = {}
        
        # should always calculate simple dists from gaussians to avoid repetitive calculations
        gaussian_fit_options = {'flucCor_switch': True, 'linear_bias_switch': True, 'debug_flag': False, 'offset_switch': True}
        self.initialize_gaussian_params(**gaussian_fit_options)
        
    def initialize_gaussian_params(self, **kwargs):
        self.dists['getAtomNumber'] = []
        self.dists['getPosX'] = []
        self.dists['getPosZ'] = []
        self.dists['getWidthX'] = []
        self.dists['getWidthZ'] = []
        self.dists['getLightCounts'] = []
        self.dists['getTimestamp'] = []
        self.dists['TOF'] = []

        index = 1
        for file in self.filelist:
            print('Processing File %d'%index)
            index += 1
            try:
                this_img = CloudImage.CloudImage(file)
                this_img_gaussian_params = this_img.getGaussianFitParams(**kwargs)
                for key in this_img_gaussian_params.keys():
                    self.dists[key].append(this_img_gaussian_params[key]) #relies on same names in this and CloudImage.py!!
                self.dists['getTimestamp'].append(this_img.getTimestamp())
                self.dists['TOF'].append(this_img.CurrTOF)
            except AttributeError:
                print('Invalid Method Name; CloudDistribution and CloudImage are out of sync!')
                raise AttributeError
            except CloudImage.FitError:
                print('Fit Error')
        
    def values(self,var, **kwargs):
        var_dist = []
        index = 1
        for file in self.filelist:
            print('Processing File %d'%index)
            index += 1
            try: # First assume it is in the variables
                this_img = CloudImage.CloudImage(file)
                this_value = this_img.getVariableValues()[var]
                # raises an AttributeError if the data is too old to have saved variables, or 
            except (KeyError, AttributeError): # Now see if it is a method name
                try:
                    this_img = CloudImage.CloudImage(file)
                    exec('this_value = this_img.' + var + '(**kwargs)')
                except AttributeError:
                    print('Invalid Method Name')
                    raise AttributeError
                except CloudImage.FitError:
                    print('Fit Error')
            var_dist.append(this_value)
        self.dists[var] = var_dist
        

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
        
    def signaltonoise(self, var):
        return self.calc_statistic(var, stats.signaltonoise)
        
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
                self.values(var)
            except AttributeError:
                print('Invalid Variable')
                return False
            return True
    
    def display_statistics(self, var, **kwargs):
        if self.does_var_exist(var, **kwargs):
            print('\nStatistics of ' + var)
            print('Mean: %2.2e'%self.mean(var))
            print('StdDev: %2.2e'%self.std(var))
            print('SNR: %2.2f'%self.signaltonoise(var))
            print('sigma_SNR: %2.2f\n'%(math.sqrt((2 + self.signaltonoise(var)**2) / len(var))))
        else:
            print('Variable does not exist!')
        
    def regression(self, var1, var2):
        if var1 not in self.dists.keys():
            print(var1 + ' distribution has not been created.')
            raise KeyError
        if var2 not in self.dists.keys():
            print(var2 + ' distribution has not been created.')
            raise KeyError
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.dists[var1], self.dists[var2])
        print('\nRegression of ' + var1 + ' against ' + var2)
        print("Slope: %2.2e"%slope)
        print("Intercept: %2.2e"%intercept)
        print("Standard Error: %2.2e"%std_err)
        print("R Value: %2.2e"%r_value)
        plt.plot(self.dists[var1],self.dists[var2],'.')
        plt.plot(self.dists[var1], slope*np.array(self.dists[var1]) + intercept)
        plt.xlabel(var1)
        plt.ylabel(var2) 
        plt.show()

        
        
if __name__ == "__main__":
    my_dists = CloudDistributions()
    
    atom_number_options =   {"axis": 1,
                            "offset_switch": True,
                            "flucCor_switch": True,
                            "debug_flag": False,
                            "linear_bias_switch": True}
    position_options =      {"flucCor_switch": True,
                            "linear_bias_switch": True}
    width_options =         {"axis": 0} #x axis
    
    # my_dists.plotdistribution('getAtomNumber',**atom_number_options)
    my_dists.display_statistics('getAtomNumber',**atom_number_options)
    # my_dists.plotdistribution('getPosX', **position_options)
    my_dists.display_statistics('getPosX', **position_options)
    # my_dists.plotdistribution('getWidthX', **width_options)
    # my_dists.display_statistics('getWidthX', **width_options)
    # my_dists.plotdistribution('getLightCounts')
    # my_dists.display_statistics('getLightCounts')
    my_dists.regression('getPosX', 'getAtomNumber')