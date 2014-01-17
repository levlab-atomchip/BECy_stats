import scipy
import scipy.io
#import parameters
import re
from scipy.optimize import curve_fit
from scipy.constants import pi, hbar
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_RE = re.compile(r'.(\d{6})\.mat')


class FitError(Exception):
    def __init__(self, statement = "Fit Error"):
        self.statement = statement

class CloudImage():
    def __init__(self,fileName):       
        self.matFile = {}
        self.fileName = fileName
        self.loadMatFile()

    def loadMatFile(self):
        scipy.io.loadmat(self.fileName,mdict = self.matFile, squeeze_me = True, struct_as_record = False)
        
        self.imageArray = self.matFile['rawImage']
        self.runDataFiles = self.matFile['runData']
        self.hfig_main = self.matFile['hfig_main']

        self.ContParName = self.runDataFiles.ContParName
        self.CurrContPar = self.runDataFiles.CurrContPar
        self.CurrTOF = self.runDataFiles.CurrTOF*1e-3

        
        self.atomImage = scipy.array(self.imageArray[:,:,0]) #scipy.array is called to make a copy, not a reference
        self.lightImage = scipy.array(self.imageArray[:,:,1])
        self.darkImage = scipy.array(self.imageArray[:,:,2])

        self.magnification = self.hfig_main.calculation.M
        self.pixel_size = self.hfig_main.calculation.pixSize
        self.imageRotation = self.hfig_main.display.imageRotation
        self.c1 = self.hfig_main.calculation.c1
        self.s_lambda = self.hfig_main.calculation.s_lambda
        self.A = self.hfig_main.calculation.A

        self.truncWinX =  self.hfig_main.calculation.truncWinX
        self.truncWinY =  self.hfig_main.calculation.truncWinY
        
        self.atomImage_trunc = self.atomImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]] 
        self.lightImage_trunc = self.lightImage[self.truncWinY[0]:self.truncWinY[-1],
                                                self.truncWinX[0]:self.truncWinX[-1]]
        self.darkImage_trunc = self.darkImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]]
                                              
        self.flucWinX = self.hfig_main.calculation.flucWinX
        self.flucWinY = self.hfig_main.calculation.flucWinY
        intAtom = np.mean(np.mean(self.atomImage[self.flucWinY[0]:self.flucWinY[-1], self.flucWinX[0]:self.flucWinX[-1]]))
        intLight = np.mean(np.mean(self.lightImage[self.flucWinY[0]:self.flucWinY[-1], self.flucWinX[0]:self.flucWinX[-1]]))
        self.flucCor = intAtom / intLight        
        return
    
    def set_fluc_corr(self, x1, x2, y1, y2):
        intAtom = np.mean(np.mean(self.atomImage[y1:y2,
                                              x1:x2]))
        intLight = np.mean(np.mean(self.lightImage[y1:y2,
                                              x1:x2]))
        self.flucCor = intAtom / intLight
    
    
    def truncate_image(self, x1, x2, y1, y2):
        self.atomImage_trunc = self.atomImage[y1:y2,x1:x2]
        self.lightImage_trunc = self.lightImage[y1:y2,x1:x2]
        self.darkImage_trunc = self.darkImage[y1:y2,x1:x2]


    def getVariablesFile(self):        
        sub_block_data = self.runDataFiles.AllFiles
        sub_blocks = {}
        for i in xrange(scipy.shape(sub_block_data)[0]):
            sub_blocks[sub_block_data[i][0]] = sub_block_data[i][1]
        return sub_blocks

    def getVariableValues(self):
        vars = self.runDataFiles.vars;
        variables_dict = {};
        for var in vars:
            variables_dict[var.name]=var.value;
        return variables_dict

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return [array[idx],idx]

    def getODImage(self, flucCor_switch = True):
        if flucCor_switch:
            ODImage = abs(np.log((self.atomImage_trunc 
                            - self.darkImage_trunc).astype(float)
                            /(self.flucCor * self.lightImage_trunc 
                            - self.darkImage_trunc).astype(float)))
        else:
            ODImage = abs(np.log((self.atomImage_trunc 
                            - self.darkImage_trunc).astype(float)
                            /(self.lightImage_trunc 
                            - self.darkImage_trunc).astype(float)))
        ODImage[np.isnan(ODImage)] = 0
        ODImage[np.isinf(ODImage)] = ODImage[~np.isinf(ODImage)].max()
        return ODImage
        
    def AtomNumber(self, axis=1, offset_switch = True, flucCor_switch = True, debug_flag = False, linear_bias_switch = True):
        ODImage = self.getODImage(flucCor_switch)
        imgcut = np.sum(ODImage,axis)
        try:
            if linear_bias_switch:
                coefs = self.fitGaussian1D(imgcut)
            else:
                coefs = self.fitGaussian1D_noline(imgcut)
        except:
            raise FitError('AtomNumber')
            # print('FitError')
            
        offset = coefs[3]
        if offset_switch:
            if linear_bias_switch:
                slope = coefs[4]
                atomNumber = self.A/self.s_lambda*(np.sum(ODImage) - 0.5*slope*len(imgcut)**2 - offset*len(imgcut))
            else:
                atomNumber = self.A/self.s_lambda*(np.sum(ODImage) - offset*len(imgcut))
        else:
            atomNumber = self.A/self.s_lambda*(np.sum(ODImage))
            
        if debug_flag:
            plt.plot(imgcut)
            params = [range(len(imgcut))]
            params.extend(coefs)
            plt.plot(self.gaussian1D(*params))
            plt.show()
        return atomNumber

    def gaussian1D(self,x,A,mu,sigma,offset, slope):
        return A*np.exp(-1.*(x-mu)**2./(2.*sigma**2.)) + offset + slope*np.array(x)
        
    def gaussian1D_noline(self,x,A,mu,sigma,offset):
        return A*np.exp(-1.*(x-mu)**2./(2.*sigma**2.)) + offset

    def gaussian2D(self,xdata, A_x,mu_x,sigma_x,A_y,mu_y,sigma_y,offset):
        x = xdata[0]
        y = xdata[1]
        return A_x*np.exp(-1.*(x-mu_x)**2./(2.*sigma_x**2.)) + A_y*np.exp(-1.*(x-mu_y)**2./(2.*sigma_y**2.)) + offset
    
    def fitGaussian1D(self,image): #fits a 1D Gaussian to a 1D image; includes constant offset and linear bias
        max_value = image.max()
        max_loc = np.argmax(image)
        [half_max,half_max_ind] = self.find_nearest(image,max_value/2.)
        hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
        p_0 = [max_value,max_loc,hwhm,0., 0.] #fit guess
        xdata = np.arange(np.size(image))
        
        coef, outmat = curve_fit(self.gaussian1D,xdata,image,p0 = p_0)
        return coef
        
    def fitGaussian1D_noline(self,image): #fits a 1D Gaussian to a 1D image; includes constant offset and linear bias
        max_value = image.max()
        max_loc = np.argmax(image)
        [half_max,half_max_ind] = self.find_nearest(image,max_value/2.)
        hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
        p_0 = [max_value,max_loc,hwhm,0.] #fit guess
        xdata = np.arange(np.size(image))
        
        coef, outmat = curve_fit(self.gaussian1D_noline,xdata,image,p0 = p_0)
        return coef

    def fitGaussian2D(self,image): #fits a 2D Gaussian to a 2D Image
        img_x = np.sum(image,0)
        img_y = np.sum(image,1)
        x_coefs= self.fitGaussian1D(img_x) #gets coefficient estimates from 1D fits
        y_coefs= self.fitGaussian1D(img_y)
        x,y = np.meshgrid(np.arange(img_x.size),np.arange(img_y.size))

        coef, outmat = curve_fit(self.gaussian2D,[x,y],image,p0 =np.delete(np.append(x_coefs,y_coefs),3))
        return coef

    def getDensity1D(self,image):
        return self.A/self.s_lambda * image

    # This method is believed to be incorrect.
    #def getBField(self,m_F = 2):
    #    image_1D = np.sum(self.getODImage(),0)
    #    density = self.getDensity1D(image_1D)
    #    return 2*hbar*self.p.w_tr*self.p.a_s/(m_F*self.p.g_F*self.p.mu_0)*density

    def getContParam(self):
        if self.ContParName == 'VOID':
            return 'void'
        else:
            Cont_Param = self.getVariableValues('Variables.m')[self.ContParName]
            return Cont_Param[self.CurrContPar]
    def getParamDefinition(self, param_name):
        return self.getVariableValues()[param_name]
    def Pos(self, axis = 0, flucCor_switch = True, linear_bias_switch = True, debug_flag = False):
        image = self.getODImage(flucCor_switch)
        imgcut = np.sum(image,axis)
        try:
            if linear_bias_switch:
                coefs = self.fitGaussian1D(imgcut)
            else:
                coefs = self.fitGaussian1D_noline(imgcut)
        except:
            raise FitError('Pos')   
            
        if debug_flag:
            plt.plot(imgcut)
            params = [range(len(imgcut))]
            params.extend(coefs)
            plt.plot(self.gaussian1D(*params))
            plt.show()
            
        return coefs[1]*self.pixel_size
    def Width(self, axis=0):
        image = self.getODImage()
        imgcut = np.sum(image,axis)
        coefs = self.fitGaussian1D(imgcut)
        return coefs[2]*self.pixel_size
    def LightCounts(self):
        return np.sum(self.lightImage - self.darkImage)
    def getChiSquared1D(self, axis = 0):
        img_1D = np.sum(self.getODImage(),axis)        
        coef = self.fitGaussian1D(img_1D)
        x = np.arange(img_1D.size)
        fit = self.gaussian1D(x,coef[0],coef[1],coef[2],coef[3])
        error = img_1D-fit
        background_1D = np.sum(self.darkImage_trunc,axis)
        variance = np.std(background_1D)**2
        chisquare = np.sum((img_1D-fit)**2)/(variance*(img_1D.size-4))
        return chisquare
        
    def getGaussianFitParams(self, flucCor_switch = False, linear_bias_switch = True, debug_flag = False, offset_switch = True):
        '''This calculates the common parameters extracted from a gaussian fit all at once, returning them in a dictionary.
        The parameters are: Atom Number, X Position, Z Position, X Width, Z Width, LightCounts'''
        ODImage = self.getODImage(flucCor_switch)
        imgcut_x = np.sum(ODImage,0)
        imgcut_z = np.sum(ODImage, 1)
        try:
            if linear_bias_switch:
                coefs_x = self.fitGaussian1D(imgcut_x)
            else:
                coefs_x = self.fitGaussian1D_noline(imgcut_x)
        except:
            # raise FitError()
            # coefs_x = [None, None, None]
            coefs_x = [0,0,0] # KLUDGE!!!
            print('Fit Error in X')
            
        try:
            if linear_bias_switch:
                coefs_z = self.fitGaussian1D(imgcut_z)
            else:
                coefs_z = self.fitGaussian1D_noline(imgcut_z)
        except:
            # raise FitError()
            print('Fit Error in Z')
        coefs_z = self.fitGaussian1D_noline(imgcut_z)
        # Using z to get atomNumber; need to add linear bias correction!
        
        offset_z = coefs_z[3]
        if offset_switch:
            atomNumber = self.A/self.s_lambda*(np.sum(ODImage) - offset_z*len(imgcut_z))
        else:
            atomNumber = self.A/self.s_lambda*(np.sum(ODImage))
            
        if debug_flag:
            plt.plot(imgcut)
            params = [range(len(imgcut))]
            params.extend(coefs)
            plt.plot(self.gaussian1D(*params))
            plt.show()
        return {'AtomNumber': atomNumber, 
                'PosX':coefs_x[1]*self.pixel_size if coefs_x[1] is not None else None, 
                'PosZ': coefs_z[1]*self.pixel_size,
                'WidthX': coefs_x[2]*self.pixel_size  if coefs_x[2] is not None else None,
                'WidthZ': coefs_z[2]*self.pixel_size,
                'LightCounts': np.sum(self.lightImage - self.darkImage),
                'Timestamp': self.Timestamp}
    def Timestamp(self):
        thisfilename = os.path.basename(self.fileName)
        return FILE_RE.search(thisfilename).group(1) #so crude!
