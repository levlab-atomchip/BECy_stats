from numpy import array
from scipy.cluster.vq import kmeans,vq
import CloudImage
from CloudImage import FitError
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

class CloudDistribution():
    def __init__(self, dir = None):
        
        
        if dir is None: # Open a windows dialog box for selecting a folder
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
            
        else:
            self.dir = dir
            
        print(self.dir)
        
        self.filelist = glob.glob(self.dir + '\\*.mat')
        self.numimgs = len(self.filelist)
        self.dists = {}
        
        # should always calculate simple dists from gaussians to avoid repetitive calculations
        gaussian_fit_options = {'flucCor_switch': False, 'linear_bias_switch': False, 'debug_flag': False, 'offset_switch': True}
        self.initialize_gaussian_params(**gaussian_fit_options)
        
    def initialize_gaussian_params(self, **kwargs):
        '''Calculate the most commonly used parameters that can be extracted from a gaussian fit'''
        self.dists['AtomNumber'] = []
        self.dists['PosX'] = []
        self.dists['PosZ'] = []
        self.dists['WidthX'] = []
        self.dists['WidthZ'] = []
        self.dists['LightCounts'] = []
        self.dists['Timestamp'] = []
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
                self.dists['Timestamp'].append(this_img.Timestamp())
                self.dists['TOF'].append(this_img.CurrTOF)
            except AttributeError:
                print('Invalid Method Name; CloudDistribution and CloudImage are out of sync!')
                raise AttributeError
            except FitError:
                print('Fit Error')
        
    def values(self,var, **kwargs):
        '''Creates a distribution for variable var, either from the variables file or by calling a CloudImage method'''
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
                # Add call to Matt's code for dealing with older data!
            var_dist.append(this_value)
        self.dists[var] = var_dist
        

    # def savevalues(self,dir,var):
        # outputfile = dir+var+ '.csv'
        # data = self.values(var)
        # with open(outputfile, 'w') as f:
            # writer = csv.writer(f)
            # for num in data[var]:
                # writer.writerow([num])

    def plot_distribution(self,var, **kwargs):
        '''Plots a histogram and time series of a distribution'''
        # numbins = np.ceil(np.power(self.numimgs,0.33))
        numbins = 20
        if self.does_var_exist(var, **kwargs):
            plt.subplot(121)
            plt.hist(self.dists[var],numbins)
            plt.ylabel('Counts')
            plt.xlabel(var)
            plt.title('Histogram of ' + var)
            plt.subplot(122)
            plt.plot(self.dists[var],marker='o',linestyle='--')
            plt.ylabel(var)
            plt.xlabel('Run Number')
            plt.title('Time Series of '+var)
            plt.show()
        else:
            print("Variable Does Not Exist")
            
    def plotdist(self, var, **kwargs):
        self.plot_distribution(var, **kwargs)
    
    
    
    def plot_gaussian_params(self):
        '''Produce various plots concerning the most commonly used parameters'''
        numbins = 20
        # numbins = np.ceil(np.power(self.numimgs,0.33))
    
        gaussian_params = ["AtomNumber", "PosX", "PosZ", "WidthsX", "WidthZ", "LightCounts"]
        plt.subplot(321)
        plt.hist(self.dists["AtomNumber"],numbins)
        plt.ylabel('Counts')
        plt.xlabel("Atom Number")
        plt.title('Number Histogram')
        plt.subplot(322)
        plt.plot(self.dists["AtomNumber"],marker='o',linestyle='--')
        plt.ylabel("Atom Number")
        plt.xlabel('Run Number')
        plt.title('Time Series')
        
        plt.subplot(323)
        plt.scatter(self.dists["PosX"], self.dists["PosZ"], marker = 'o')
        plt.ylabel('Z Position')
        plt.xlabel('X Position')
        plt.title('Location of Cloud Center')
        plt.subplot(324)
        plt.scatter(self.dists["WidthX"], self.dists["WidthZ"], marker = 'o')
        plt.ylabel("Z Width")
        plt.xlabel("X Width")
        plt.title("Cloud Widths")
        
        plt.subplot(325)
        plt.hist(self.dists["LightCounts"], numbins)
        plt.ylabel('Counts')
        plt.xlabel('Light Counts')
        plt.title('Light Intensity Distribution')
        
        plt.subplot(326)
        plt.scatter(self.dists["LightCounts"], self.dists["AtomNumber"], marker = 'o')
        plt.xlabel("Light Counts")
        plt.ylabel("Atom Number")
        plt.title("Atom Number vs. Light Intensity")
        
        plt.tight_layout()
        plt.show()
        
    
    def mean(self,var):
        return self.calc_statistic(var, np.mean)
        
    def std(self,var):
        return self.calc_statistic(var, np.std)
    
    def median(self,var):
        return self.calc_statistic(var, np.median)
        
    def signaltonoise(self, var):
        return self.calc_statistic(var, stats.signaltonoise)
        
    def snr(self, var):
        return self.signaltonoise(var)
        
    def calc_statistic(self, var, statistic):
        '''Returns the value of statistic for the given variable'''
        if self.does_var_exist(var):
            return statistic(self.dists[var])
        else:
            print("Variable Does Not Exist")
            return Null
            
    def calcstat(self, var, statistic):
        return self.calc_statistic(var, statistic)
            
    def does_var_exist(self, var, **kwargs):
        '''Checks to see if the variable has a distribution defined. If not, attempts to make one.'''
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
        '''Display several commonly used statistics'''
        if self.does_var_exist(var, **kwargs):
            print('\nStatistics of ' + var)
            print('Mean: %2.2e'%self.mean(var))
            print('StdDev: %2.2e'%self.std(var))
            print('SNR: %2.2f'%self.signaltonoise(var))
            print('sigma_SNR: %2.2f\n'%(math.sqrt((2 + self.signaltonoise(var)**2) / len(var))))
        else:
            print('Variable does not exist!')
            
    def dispstat(self, var, **kwargs):
        return self.display_statistics(var, **kwargs)
        
    def regression(self, var1, var2):
        '''Perform linear regression on two variables and plot the result'''
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

    def kmeans(self, var1, var2, num_clusters=2):
        '''Perform a common clustering algorithm'''
        if var1 not in self.dists.keys():
            print(var1 + ' distribution has not been created.')
            raise KeyError
        if var2 not in self.dists.keys():
            print(var2 + ' distribution has not been created.')
            raise KeyError
        # data generation
        data = np.transpose(array([self.dists[var1], self.dists[var2]]))

        # computing K-Means with K = num_clusters
        centroids,_ = kmeans(data,num_clusters)
        # assign each sample to a cluster
        idx,_ = vq(data,centroids)

        # some plotting using numpy's logical indexing
        plt.plot(data[idx==0,0],data[idx==0,1],'ob',
             data[idx==1,0],data[idx==1,1],'or')
        plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
        plt.show()


if __name__ == "__main__":
    dir = r"C:\Users\Will\Desktop\Levlab Projects\BECy Data\MacCap Num 2014-01-16"
    my_dists = CloudDistribution(dir)
    
    # atom_number_options =   {"axis": 1,
                            # "offset_switch": True,
                            # "flucCor_switch": True,
                            # "debug_flag": False,
                            # "linear_bias_switch": True}
    # position_options =      {"flucCor_switch": True,
                            # "linear_bias_switch": True}
    # width_options =         {"axis": 0} #x axis
    
    # my_dists.plot_distribution('AtomNumber',**atom_number_options)
    # my_dists.display_statistics('AtomNumber',**atom_number_options)
    # my_dists.plot_distribution('PosX', **position_options)
    # my_dists.display_statistics('PosX', **position_options)
    # my_dists.plot_distribution('WidthX', **width_options)
    # my_dists.display_statistics('WidthX', **width_options)
    # my_dists.plot_distribution('LightCounts')
    # my_dists.display_statistics('LightCounts')
    # my_dists.regression('PosX', 'AtomNumber')
    
    # my_dists.plot_gaussian_params()
    # my_dists.regression('PosX', 'WidthX')
    # my_dists.regression('PosX', 'AtomNumber')
    # my_dists.regression('WidthX', 'AtomNumber')
    # my_dists.plot_distribution('WidthX')
    # my_dists.display_statistics('WidthX')
    
    # my_dists.plot_distribution('PosX')
    # my_dists.display_statistics('PosX')
    my_dists.kmeans('PosX', 'AtomNumber', 2)