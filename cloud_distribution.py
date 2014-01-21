'''This is a class definition for getting distributional
    information over a set of cloud images
Right now this can provide the functionality of:
numberhist, posdist, widthdist, intensitydist, numpos,
    paramregress, pos_regress, posxvy, twoparamregress'''

from numpy import array
from scipy.cluster.vq import kmeans, vq
import cloud_image
from cloud_image import FitError
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import stats
import math

import win32gui
from win32com.shell import shell, shellcon

DEBUG_FLAG = False
LINEAR_BIAS_SWITCH = True

class CloudDistribution(object):

    '''class representing distributions of parameters over many images'''

    def __init__(self, directory=None):

        # Open a windows dialog box for selecting a folder
        if directory is None:
            desktop_pidl = shell.SHGetFolderLocation(0,
                        shellcon.CSIDL_DESKTOP, 0, 0)
            pidl, _, _ = shell.SHBrowseForFolder(
                win32gui.GetDesktopWindow(),
                desktop_pidl,
                "Choose a folder",
                0,
                None,
                None
            )
            self.directory = shell.SHGetPathFromIDList(pidl)

        else:
            self.directory = directory

        print self.directory

        self.filelist = glob.glob(self.directory + '\\*.mat')
        self.numimgs = len(self.filelist)
        self.dists = {}

        # should always calculate simple dists from gaussians
        # to avoid repetitive calculations
        gaussian_fit_options = {'fluc_cor_switch': False,
                                'linear_bias_switch': LINEAR_BIAS_SWITCH,
                                'debug_flag': DEBUG_FLAG,
                                'offset_switch': True}
        self.initialize_gaussian_params(**gaussian_fit_options)

    def initialize_gaussian_params(self, **kwargs):
        '''Calculate the most commonly used parameters
        that can be extracted from a gaussian fit'''
        self.dists['atom_number'] = []
        self.dists['position_x'] = []
        self.dists['position_z'] = []
        self.dists['width_x'] = []
        self.dists['width_z'] = []
        self.dists['light_counts'] = []
        self.dists['timestamp'] = []
        self.dists['tof'] = []

        index = 1
        for this_file in self.filelist:
            print 'Processing File %d' % index
            index += 1
            try:
                this_img = cloud_image.CloudImage(this_file)
                this_img_gaussian_params = \
                    this_img.get_gaussian_fit_params(**kwargs)
                for key in this_img_gaussian_params.keys():
                    try:
                        self.dists[key].append(this_img_gaussian_params[key])
                    except AttributeError:
                        print '''Invalid Method Name %s;
                        CloudDistribution and CloudImage are out of sync!'''%key
                        raise AttributeError
                    # relies on same names in this and CloudImage.py!!
                self.dists['timestamp'].append(this_img.timestamp())
                self.dists['tof'].append(this_img.curr_tof)

            except FitError:
                print 'Fit Error'

    def values(self, var, **kwargs):
        '''Creates a distribution for variable var, either
        from the variables file or by calling a CloudImage method'''
        var_dist = []
        index = 1
        for this_file in self.filelist:
            print 'Processing File %d' % index
            index += 1
            try:  # First assume it is in the variables
                this_img = cloud_image.CloudImage(this_file)
                this_value = this_img.get_variables_values()[var]
                # raises an AttributeError if the data is
                # too old to have saved variables, or
            # Now see if it is a method name
            except (KeyError, AttributeError):
                try:
                    this_img = cloud_image.CloudImage(this_file)
                    exec('this_value = this_img.' + var + '(**kwargs)')
                except AttributeError:
                    print 'Invalid Method Name'
                    raise AttributeError
                except cloud_image.FitError:
                    print 'Fit Error'
                # Add call to Matt's code for dealing with older data!
            var_dist.append(this_value)
        self.dists[var] = var_dist

    def plot_distribution(self, var, **kwargs):
        '''Plots a histogram and time series of a distribution'''
        # numbins = np.ceil(np.power(self.numimgs,0.33))
        numbins = 20
        if self.does_var_exist(var, **kwargs):
            plt.subplot(121)
            plt.hist(self.dists[var], numbins)
            plt.ylabel('Counts')
            plt.xlabel(var)
            plt.title('Histogram of ' + var)
            plt.subplot(122)
            plt.plot(self.dists[var], marker='o', linestyle='--')
            plt.ylabel(var)
            plt.xlabel('Run Number')
            plt.title('Time Series of ' + var)
            plt.show()
        else:
            print "Variable Does Not Exist"

    def plotdist(self, var, **kwargs):
        '''alias for plot_distribution'''
        self.plot_distribution(var, **kwargs)

    def plot_gaussian_params(self):
        '''Produce various plots concerning the most commonly used parameters'''
        numbins = 20
        # numbins = np.ceil(np.power(self.numimgs,0.33))

        plt.subplot(321)
        plt.hist(self.dists["atom_number"], numbins)
        plt.ylabel('Counts')
        plt.xlabel("Atom Number")
        plt.title('Number Histogram')
        plt.subplot(322)
        plt.plot(self.dists["atom_number"], marker='o', linestyle='--')
        plt.ylabel("Atom Number")
        plt.xlabel('Run Number')
        plt.title('Time Series')

        plt.subplot(323)
        plt.scatter(self.dists["position_x"],
                        self.dists["position_z"], marker='o')
        plt.ylabel('Z Position')
        plt.xlabel('X Position')
        plt.title('Location of Cloud Center')
        plt.subplot(324)
        plt.scatter(self.dists["width_x"], self.dists["width_z"], marker='o')
        plt.ylabel("Z Width")
        plt.xlabel("X Width")
        plt.title("Cloud Widths")

        plt.subplot(325)
        plt.hist(self.dists["light_counts"], numbins)
        plt.ylabel('Counts')
        plt.xlabel('Light Counts')
        plt.title('Light Intensity Distribution')

        plt.subplot(326)
        plt.scatter(self.dists["light_counts"],
                    self.dists["atom_number"], marker='o')
        plt.xlabel("Light Counts")
        plt.ylabel("Atom Number")
        plt.title("Atom Number vs. Light Intensity")

        plt.tight_layout()
        plt.show()

    def mean(self, var):
        return self.calc_statistic(var, np.mean)

    def std(self, var):
        return self.calc_statistic(var, np.std)

    def median(self, var):
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
            print "Variable Does Not Exist"
            return Null

    def calcstat(self, var, statistic):
        return self.calc_statistic(var, statistic)

    def does_var_exist(self, var, **kwargs):
        '''Checks to see if the variable has a distribution defined.
        If not, attempts to make one.'''
        if var in self.dists.keys():
            return True
        else:
            try:
                self.values(var, **kwargs)
            except AttributeError:
                print 'Invalid Variable'
                return False
            return True

    def display_statistics(self, var, **kwargs):
        '''Display several commonly used statistics'''
        if self.does_var_exist(var, **kwargs):
            print '\nStatistics of ' + var
            print 'Mean: %2.2e' % self.mean(var)
            print 'StdDev: %2.2e' % self.std(var)
            print 'SNR: %2.2f' % self.signaltonoise(var)
            print 'sigma_SNR: %2.2f\n' % (math.sqrt((2 +
                    self.signaltonoise(var) ** 2) / len(var)))
        else:
            print 'Variable does not exist!'

    def dispstat(self, var, **kwargs):
        return self.display_statistics(var, **kwargs)

    def regression(self, var1, var2):
        '''Perform linear regression on two variables and plot the result'''
        if var1 not in self.dists.keys():
            print var1 + ' distribution has not been created.'
            raise KeyError
        if var2 not in self.dists.keys():
            print var2 + ' distribution has not been created.'
            raise KeyError
        slope, intercept, r_value, _, std_err = \
            stats.linregress(self.dists[var1], self.dists[var2])
        print '\nRegression of ' + var1 + ' against ' + var2
        print "Slope: %2.2e" % slope
        print "Intercept: %2.2e" % intercept
        print "Standard Error: %2.2e" % std_err
        print "R Value: %2.2e" % r_value
        plt.plot(self.dists[var1], self.dists[var2], '.')
        plt.plot(
            self.dists[var1],
            slope * np.array(self.dists[var1]) + intercept)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.show()

    def kmeans(self, var1, var2, num_clusters=2):
        '''Perform a common clustering algorithm'''
        if var1 not in self.dists.keys():
            print var1 + ' distribution has not been created.'
            raise KeyError
        if var2 not in self.dists.keys():
            print var2 + ' distribution has not been created.'
            raise KeyError
        # data generation
        data = np.transpose(array([self.dists[var1], self.dists[var2]]))

        # computing K-Means with K = num_clusters
        centroids, _ = kmeans(data, num_clusters)
        # assign each sample to a cluster
        idx, _ = vq(data, centroids)

        # some plotting using numpy's logical indexing
        plt.plot(data[idx == 0, 0], data[idx == 0, 1], 'ob',
                 data[idx == 1, 0], data[idx == 1, 1], 'or')
        plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
        plt.show()


if __name__ == "__main__":
    MY_DISTS = CloudDistribution()

    # atom_number_options =   {"axis": 1,
                            # "offset_switch": True,
                            # "flucCor_switch": True,
                            # "debug_flag": False,
                            # "linear_bias_switch": True}
    # position_options =      {"flucCor_switch": True,
                            # "linear_bias_switch": True}
    # width_options =         {"axis": 0} #x axis

    # MY_DISTS.plot_distribution('atom_number',**atom_number_options)
    # MY_DISTS.display_statistics('atom_number',**atom_number_options)
    # MY_DISTS.plot_distribution('position_x', **position_options)
    # MY_DISTS.display_statistics('position_x', **position_options)
    # MY_DISTS.plot_distribution('width_x', **width_options)
    # MY_DISTS.display_statistics('width_x', **width_options)
    # MY_DISTS.plot_distribution('light_counts')
    # MY_DISTS.display_statistics('light_counts')
    # MY_DISTS.regression('position_x', 'atom_number')

    # MY_DISTS.plot_gaussian_params()
    # MY_DISTS.regression('position_x', 'width_x')
    # MY_DISTS.regression('position_x', 'atom_number')
    # MY_DISTS.regression('width_x', 'atom_number')
    # MY_DISTS.plot_distribution('width_x')
    # MY_DISTS.display_statistics('width_x')

    # MY_DISTS.plot_distribution('position_x')
    # MY_DISTS.display_statistics('position_x')
    MY_DISTS.kmeans('position_x', 'atom_number', 2)
