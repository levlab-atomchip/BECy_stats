'''Cloud Distributions

This is a class definition for getting distributional
information over a set of cloud images in a single directory.'''

import fittemp
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import curve_fit
import cloud_image
from cloud_image import FitError
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import stats
import math
import hempel
import pprint
import fit_double_gaussian as fdg
import platform
from BECphysics import M, KB, GRAVITY
from fit_functions import temp_func, lifetime_func, freq_func, magnif_func

# Flags for setting module behavior
DEBUG_FLAG = False                  #Debug mode; shows each fit
LINEAR_BIAS_SWITCH = False          #Use linear bias in gaussian fits
FLUC_COR_SWITCH = True             #Use fluctuation correction
OFFSET_SWITCH = True                #Use fit to offset densities for number calculation
FIT_AXIS = 1;                       #0 is x, 1 is z      
CUSTOM_FIT_SWITCH = False            #Use CUSTOM_FIT_WINDOW
USE_FIRST_WINDOW = False            #Use the fit window from the first image for all images
PIXEL_UNITS = False                  #Return lengths and positions in pixels
DOUBLE_GAUSSIAN = False               #Fit a double gaussian
DEBUG_DOUBLE = False                #Debug mode for double gaussian fits
OVERLAP = False                      #True if the data is actually two gaussians  overlapping and need to be fit to one gaussian
if DOUBLE_GAUSSIAN:
    OVERLAP = False         #always fit single gaussian if the two gaussians overlap

#CUSTOM_FIT_WINDOW = [393,623,145,163]   #x0, x1, y0, y1, no atoms
CUSTOM_FIT_WINDOW = [393,623,158,176]   #x0, x1, y0, y1, unperturbed at 40 A

CAMPIXSIZE = 3.75e-6 #m, physical size of camera pixel
cloud_width = 1.0*10**-6.0 #used in OVERLAP, assuming the overlapping gaussians both have the same sigma of 1um
           

class CloudDistribution(object):
    '''class representing distributions of parameters over many images'''

    def __init__(self, directory=None, INITIALIZE_GAUSSIAN_PARAMS=True):

        self.directory = directory
        self.INITIALIZE_GAUSSIAN_PARAMS = INITIALIZE_GAUSSIAN_PARAMS

        print self.directory

        # Find all .mat files
        if platform.system()=='Darwin' or platform.system()=='Linux':
            #this step checks if we are running in Mac OS and change the format of the directory
            self.filelist = sorted(glob.glob(self.directory + '*.mat')) #just a MAC OS directory difference
        else:
            self.filelist = sorted(glob.glob(self.directory + '\\*.mat'))
        self.numimgs = len(self.filelist)
        self.dists = {}
        self.outliers = {}
        self.cont_par_name = None
        
        self.custom_fit_window = CUSTOM_FIT_WINDOW
        
        # should always calculate simple dists from gaussians
        # to avoid repetitive calculations
        self.gaussian_fit_options = {'fluc_cor_switch': FLUC_COR_SWITCH,
                                'linear_bias_switch': LINEAR_BIAS_SWITCH,
                                'debug_flag': DEBUG_FLAG,
                                'offset_switch': OFFSET_SWITCH,
                                'fit_axis': FIT_AXIS,
                                'custom_fit_switch': CUSTOM_FIT_SWITCH,
                                'use_first_window': USE_FIRST_WINDOW,
                                'pixel_units': PIXEL_UNITS}
        if self.INITIALIZE_GAUSSIAN_PARAMS:
            print("Initializing Gaussian Parameters")
            self.initialize_gaussian_params(**self.gaussian_fit_options)
        
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
        
        if OVERLAP:
            self.dists['d_peaks']=[] #inferred distance between two gaussians that are overlapping
            self.dists['h_from_sample']=[] #height of atoms from sample
        
        if DOUBLE_GAUSSIAN:
            self.dists['d_peaks']=[] #distance between the two gaussian peaks
            self.dists['position_1']=[]#position of first peak
            self.dists['position_2']=[]#position of second peak
            self.dists['sigma_1']=[]#width of first peak
            self.dists['sigma_2']=[]#width of second peak
            self.dists['sample_position']=[]#position of the sample, i.e. mid point of position_1 and position_2

        index = 1
        if DOUBLE_GAUSSIAN:
            #p_0=fdg.fit_double_gaussian_1d(self.filelist[0],True)
            p_0= [20.,20.,21.,34.,3.5,3.2,288.,-2.9] # guess params for double gaussian fit in pixels or OD 
            #[amplitude of 1st peak, amplitude of 2nd peak, position_1, position_2, sigma_1, sigma_2,offset,slope]
        for this_file in self.filelist:
            if USE_FIRST_WINDOW and index == 1:
                first_img = cloud_image.CloudImage(this_file)
                self.custom_fit_window = [first_img.trunc_win_x[0],
                                     first_img.trunc_win_x[-1],
                                     first_img.trunc_win_y[0],
                                     first_img.trunc_win_y[-1]]
                
            print 'Processing File %d' % index
            index += 1
            if not DOUBLE_GAUSSIAN:
                try:
                    this_img_gaussian_params = \
                                self.get_gaussian_params(this_file, **kwargs)

                    for key in this_img_gaussian_params.keys():
                        try:
                            self.dists[key].append(this_img_gaussian_params[key])
                        except AttributeError:
                            print '''Invalid Method Name %s;
                            CloudDistribution and CloudImage are out of sync!'''%key
                            raise AttributeError
                        # relies on same names in this and CloudImage.py!!

                except FitError:
                    print 'Fit Error'
                
                
                        
            else:
                #fit data to double gaussians
                print "Processing " + this_file
                self.get_double_gaussian_params(this_file,p_0)
            
        if OVERLAP:
                
                for i in self.dists['width_z']:
                    self.dists['d_peaks'].append(2.0*(1/i*np.sqrt((i**4.0-cloud_width**4.0)/2.0)))
                    self.dists['h_from_sample'].append(1/i*np.sqrt((i**4.0-cloud_width**4.0)/2.0))
                

    def get_gaussian_params(self, file, **kwargs):
        '''return cloud parameters extracted from gaussian fits'''
        this_img = cloud_image.CloudImage(file)
        if CUSTOM_FIT_SWITCH:
            this_img.truncate_image(*self.custom_fit_window)
        gaussian_params = \
                    this_img.get_gaussian_fit_params(**kwargs)
        gaussian_params['timestamp'] = this_img.timestamp()
        gaussian_params['tof'] = this_img.curr_tof
        return gaussian_params


    def control_param_dist(self):
        '''Create a distribution for the control parameter'''
        self.cont_par_name = None
        index = 1
        cont_pars = []
        for this_file in self.filelist:
            print 'Processing File %d' % index
            index += 1

            this_img = cloud_image.CloudImage(this_file)
            this_img_cont_par_name = this_img.cont_par_name
            if self.cont_par_name is None:
                self.cont_par_name = this_img_cont_par_name
            else:
                if this_img_cont_par_name != self.cont_par_name:
                    print 'No single control parameter!'
                    self.cont_par_name = None
                    raise Exception
            cont_pars.append(this_img.curr_cont_par)
        self.dists[self.cont_par_name] = cont_pars

    def temperature_groups(self):
        '''Create a list of lists of images in the same temperature set'''
        # This method assumes without warrant that the images are sorted
        # by timestamp in each distribution. Does that make me a bad person?
        tempseq = []
        seqs = []
        last_TOF = -1 # All valid TOFs are larger than this.
        for index, this_TOF in enumerate(self.dists['tof']):
            if this_TOF < last_TOF:
                seqs.append(tempseq)
                tempseq = []
                tempseq.append(index)
            else:
                tempseq.append(index)
            last_TOF = this_TOF
        seqs.append(tempseq)
        self.dists['temperature_groups'] = seqs

    def temp_dist(self):
        '''Calculate temperatures, given temperature groups'''
        # code for checking that temp groups exists needed
        self.temperature_groups()
        temp_x = []
        temp_z = []
        for temp_group in self.dists['temperature_groups']:
            this_widths_x = [self.dists['width_x'][index]
                                for index in temp_group]
            this_widths_z = [self.dists['width_z'][index]
                                for index in temp_group]
            this_tofs = [self.dists['tof'][index] for index in temp_group]
            this_temp_x, _ = fittemp.fittemp(this_tofs, this_widths_x)
            this_temp_z, _ = fittemp.fittemp(this_tofs, this_widths_z)
            temp_x.append(this_temp_x)
            temp_z.append(this_temp_z)
        self.dists['temp_x'] = temp_x
        self.dists['temp_z'] = temp_z


    def values(self, var, **kwargs):
        '''Create a distribution for variable var, either
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

    def find_outliers(self, var, nMADM):
        '''Add an entry to the outliers dictionary for the given variable,
        of the form {var : [list of outlier indices]}'''
        _, bad_indices = hempel.hempel_filter(self.dists[var], nMADM)
        self.outliers[var] = bad_indices

    def remove_outliers(self, var, nMADM = 4):
        '''Remove entries from all distributions for which the given
        variable is an outlier.'''
        if not self.does_var_exist(var):
            print '%s does not exist!'%var
            return
        if var not in self.outliers.keys():
            self.find_outliers(var, nMADM)
        bad_indices = self.outliers[var]
        for variable in self.dists.keys():
            temp_dist = [j for i, j in enumerate(self.dists[variable])
                            if i not in bad_indices]
            self.dists[variable] = temp_dist

    def does_var_exist(self, var, **kwargs):
        '''Check to see if the variable has a distribution defined.
        If not, attempt to make one.'''
        if var in self.dists.keys():
            return True
        else:
            try:
                self.values(var, **kwargs)
            except AttributeError:
                print 'Invalid Variable'
                return False
            return True

    def plot_distribution(self, var, **kwargs):
        '''Plot a histogram and time series of a distribution'''
        # numbins = np.ceil(np.power(self.numimgs,0.33)) #This is recommended in the statistics literature but proves unsatisfying
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
        
    def allan_dev(self, var):
        return self.calc_statistic(var, lambda x: math.sqrt(0.5*np.mean(np.diff(x)**2)))

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
        '''Check to see if the variable has a distribution defined.
        If not, attempt to make one.'''
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
        #TODO: make this create a dictionary instead
        if self.does_var_exist(var, **kwargs):
            print '\n'+self.directory
            print '\nStatistics of ' + var
            print 'Mean: %2.2e' % self.mean(var)
            print 'StdDev: %2.2e' % self.std(var)
            print 'SNR: %2.2f' % self.signaltonoise(var)
            print 'sigma_SNR: %2.2f' % (math.sqrt((2 +
                    self.signaltonoise(var) ** 2) / len(self.dists[var])))
            print 'Allan Deviation: %2.2e' % self.allan_dev(var)
            print 'Allan SNR: %2.2f\n' %  (self.mean(var) / self.allan_dev(var))
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self.gaussian_fit_options)
        else:
            print 'Variable does not exist!'

    def dispstat(self, var, **kwargs):
        '''alias for display_statistics'''
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
        print "R^2 Value: %2.2e" % r_value**2
        plt.plot(self.dists[var1], self.dists[var2], '.')
        plt.plot(
            self.dists[var1],
            slope * np.array(self.dists[var1]) + intercept)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.show()
        
    def lifetime(self):
        '''Perform exponential regression to determine the lifetime'''
        if self.cont_par_name not in self.dists.keys():
            self.control_param_dist()
        p0 = np.array([self.dists['atom_number'][0], 0.2, 0]) 
        popt, pcov = curve_fit(lifetime_func, np.array(self.dists[self.cont_par_name]), np.array(self.dists['atom_number']), p0)
        print '\nRegression of atom number against ' + self.cont_par_name
        print 'Lifetime: %2.1f' % (1/popt[1])
        print 'sigma: %2.1e' % np.sqrt(pcov[1][1])
        plt.plot(self.dists[self.cont_par_name], self.dists['atom_number'], '.')
        time_srtd = sorted(self.dists[self.cont_par_name])
        time_array = np.linspace(time_srtd[0], time_srtd[-1], 100)
        # num_srtd = [x for (y, x) in sorted(zip(self.dists[self.cont_par_name], self.dists['atom_number']))]
        plt.plot(
            time_array,
            lifetime_func(np.array(time_array), *popt))
        plt.xlabel(self.cont_par_name)
        plt.ylabel('Atom Number')
        plt.show()
    
    def trap_freq(self, axis=1):
        '''Fit a position to a sine function to determine trap frequency'''
        if self.cont_par_name not in self.dists.keys():
            self.control_param_dist()
        
        times = np.array(self.dists[self.cont_par_name])
        if axis == 0:
            positions = np.array(self.dists['position_x'])
            pguess = np.array([2*math.pi*10, np.max(positions), 0, 0])
        else:
            positions = np.array(self.dists['position_z'])
            pguess = np.array([2*math.pi*700, np.max(positions), 0, 0])

        popt, pcov = curve_fit(freq_func, times, positions, pguess)
        
        print '\nRegression of position against ' + self.cont_par_name
        print("Frequency: %2.2f Hz"%(popt[0] / 2 / math.pi))
        print("Sigma: %2.2f Hz"%math.sqrt(pcov[0][0]))

        plt.plot(times, positions, '.')
        time_axis = np.linspace(np.min(times), np.max(times))
        plt.plot(time_axis, freq_func(time_axis, *popt))
        plt.xlabel(self.cont_par_name)
        if axis == 0:
            plt.ylabel('X Position')
        else:
            plt.ylabel('Z Position')
        plt.show()

    def magnification(self):
        '''Extract magnification by fitting to a parabola'''
        T = np.array(self.dists['tof'])
        Y = np.array(self.dists['position_z'])
        Y = np.max(Y)-Y
        p0 = np.array([GRAVITY*3.0 / (2.0 * CAMPIXSIZE), 0, np.min(Y)])
        popt, pcov = curve_fit(magnif_func, T, Y, p0)
        M = 2*popt[0] * CAMPIXSIZE / GRAVITY
        sigma = 2*np.sqrt(pcov[0][0]) * CAMPIXSIZE / GRAVITY
        print "Magnification: %2.2f"%M
        print "Sigma: %2.2f"%sigma
        plt.plot(T, Y, '.')
        xax = np.linspace(np.min(T), np.max(T))
        plt.plot(xax, magnif_func(xax, *popt))
        plt.xlabel('TOF / ms')
        plt.ylabel('Height / px')
        plt.title('Magnification Fit')
        plt.show()
        
    def temperature(self, axis = 1):
        '''Extract temperature'''
        T = np.array(self.dists['tof'])
        if axis == 0:
            S = np.array(self.dists['width_x'])
        else:
            S = np.array(self.dists['width_z'])
        p0 = np.array([np.min(S), 0.002])
        tempfit_params, covars = curve_fit(temp_func, T, S, p0)
        sigma_v = tempfit_params[1]
        temp = M * sigma_v**2 / KB
        temp_var = 4.0 * M * temp * covars[1][1] / KB
        print "Temperature: %2.2f nK"%(temp*1e9)
        print "Sigma: %2.2f nK"%(np.sqrt(temp_var)*1e9)
        plt.plot(T, S, '.')
        xax = np.linspace(np.min(T), np.max(T))
        plt.plot(xax, temp_func(xax, *tempfit_params))
        plt.xlabel('TOF / ms')
        plt.ylabel('Width / px')
        plt.title('Temperature Fit')
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
        data = np.transpose(np.array([self.dists[var1], self.dists[var2]]))

        # computing K-Means with K = num_clusters
        centroids, _ = kmeans(data, num_clusters)
        # assign each sample to a cluster
        idx, _ = vq(data, centroids)

        # some plotting using numpy's logical indexing
        plt.plot(data[idx == 0, 0], data[idx == 0, 1], 'ob',
                 data[idx == 1, 0], data[idx == 1, 1], 'or')
        plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
        plt.show()
        
    def get_average_image(self, **kwargs):
        '''Return the average of all odimages in a distribution'''
        firstimg = cloud_image.CloudImage(self.filelist[0])
        if CUSTOM_FIT_SWITCH:
                firstimg.truncate_image(*self.custom_fit_window)
        avg_img = np.zeros(np.shape(firstimg.get_od_image(**kwargs)))
    
        for this_file in self.filelist:
            this_img = cloud_image.CloudImage(this_file)
            print this_img.filename
            if CUSTOM_FIT_SWITCH:
                this_img.truncate_image(*self.custom_fit_window)
            this_odimg = this_img.get_od_image(**kwargs)
            avg_img += this_odimg
        
        avg_img = avg_img / len(self.filelist)
        self.avg_img = avg_img
        
        return avg_img
        
    def get_snr_map(self):
        '''return a map of SNR of density at each pixel'''
        if not hasattr(self, 'avg_img'):
            self.get_average_image()
        var_img = np.zeros(np.shape(self.avg_img))
        
        for this_file in self.filelist:
            this_img = cloud_image.CloudImage(this_file).get_od_image()
            var_img += (this_img - self.avg_img)**2
            
        var_img /= self.numimgs
        
        self.var_img = var_img
        self.std_img = np.sqrt(var_img)
        self.snr_map = self.avg_img / self.std_img
        
        return self.snr_map
    
    def fit_double_gaussian(self, file, p_0_guess = None):
        '''fits one data file to a double-gaussian'''
        data = np.sum(np.array(cloud_image.CloudImage(file).get_od_image()),1)
        try:
            if p_0_guess == None:
                coef=fdg.fit_double_gaussian_1d(data,True)
            else:
                coef=fdg.fit_double_gaussian_1d(data,False,p_0_guess)
            if DEBUG_DOUBLE:

                xdata = np.arange(np.size(data))
                plt.plot(fdg.double_gaussian_1d(xdata,*coef))
                plt.plot(data)
                print coef

                plt.show()
        except FitError:
            print 'There maybe a fit error for fitting double gaussian.'
            
        
        return coef
    
    def get_double_gaussian_params(self,file,p_0_guess=None):
        '''get parameters and store them in the correct dictionaries'''
        coef=self.fit_double_gaussian(file,p_0_guess)
        double_gaussian_params=np.array(['d_peaks','position_1','position_2','sigma_1','sigma_2','sample_position'])
        for key in double_gaussian_params:
            self.dists[key].append(self.calc_double_gaussian_params(coef,key))
    
    def calc_double_gaussian_params(self,coef,key):
        '''calculate the gaussian parameters from the fit coeficients'''
        if key == 'd_peaks':
            return np.abs(coef[2]-coef[3])
        elif key == 'position_1':
            return np.min(coef[2:4])
        elif key == 'position_2':
            return np.max(coef[2:4])
        elif key == 'sigma_1':
            return coef[4]
        elif key == 'sigma_2':
            return coef[5]
        elif key == 'sample_position':
            return (coef[2]+coef[3])/2.0
        else:
            print 'Incorrect key: parameter may not exist for fitting double gaussian.'
    
    def get_field_noise(self, filter_on=False):
        fas = []
        for ii, this_file in enumerate(self.filelist):
            print 'Processing file %d'%(ii+1)
            this_img = cloud_image.CloudImage(this_file)
            if CUSTOM_FIT_SWITCH:
                print 'Using Custom Window'
                this_img.truncate_image(*self.custom_fit_window)
            this_field = this_img.get_gerbier_field(filter_on)
            this_field -= np.min(this_field)
            fas.append(this_field)
        fa_arr = np.vstack(tuple(fas))
        fa_std = np.std(fa_arr, axis=0)
        return fa_std
        
    def get_saturation(self):
        sats = []
        for ii, this_file in enumerate(self.filelist):
            print 'Processing file %d'%(ii+1)
            this_img = cloud_image.CloudImage(this_file)
            if CUSTOM_FIT_SWITCH:
                print 'Using Custom Window'
                this_img.truncate_image(*self.custom_fit_window)
            this_saturation = this_img.saturation()
            sats.append(this_saturation)
        return np.mean(sats)
 

CD = CloudDistribution
        
if __name__ == "__main__":
    pass
