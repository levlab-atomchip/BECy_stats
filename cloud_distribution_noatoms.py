from cloud_distribution import CloudDistribution
from no_atom_image import NoAtomImage as nai
import numpy as np

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

CUSTOM_FIT_WINDOW = [393,623,130,148]   #x0, x1, y0, y1
CAMPIXSIZE = 3.75e-6 #m, physical size of camera pixel
cloud_width = 1.0*10**-6.0 #used in OVERLAP, assuming the overlapping gaussians both have the same sigma of 1um

class CloudDistributionNoAtoms(CloudDistribution):
    def __init__(self, directory=None, INITIALIZE_GAUSSIAN_PARAMS=True):
        CloudDistribution.__init__(self,directory, INITIALIZE_GAUSSIAN_PARAMS)
        self.makeimage = nai
