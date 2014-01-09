import scipy
import scipy.io
#import parameters
import re
from scipy.optimize import curve_fit
from scipy.constants import pi, hbar
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======

>>>>>>> added linear bias to fits

class CloudImage():
    def __init__(self,fileName,p=None):
        if p==None:
            pass
        #self.p = parameters.bfield_parameters()
        else:
            pass
            #self.p=p
        
        self.matFile = {}
        self.fileName = fileName
        self.loadMatFile()

        #we need a control parameter to group images
        #we need a location variable
        #we need a way to search for variabgles in the image file
        
        # self.hfig_main
        # self.imageArray
        # self.runDataFiles
        # self.atomImage
        # self.lightImage
        # self.darkImage
        
        # self.magnification
        # self.pixel_size
        # self.imageRotation
        # self.c1
        # self.s_lambda
        # self.A

    def loadMatFile(self):
        #scipy.io.loadmat(self.fileName,mdict = self.matFile)
        scipy.io.loadmat(self.fileName,mdict = self.matFile, squeeze_me = True, struct_as_record = False)
        
        self.imageArray = self.matFile['rawImage']
        self.runDataFiles = self.matFile['runData']
        self.hfig_main = self.matFile['hfig_main']

        #self.ContParName = self.runDataFiles[0,0][11][0]
        #self.CurrContPar = self.runDataFiles[0,0][12][0]
        #self.CurrTOF = self.runDataFiles[0,0][13][0]
        self.ContParName = self.runDataFiles.ContParName
        # print(self.ContParName)
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

        #self.magnification = self.hfig_main[0][0][85][0][0][2]
        #self.pixel_size = self.hfig_main[0][0][85][0][0][1]
        #self.imageRotation = self.hfig_main[0][0][86][0][0][0][0]
        #self.c1 = self.hfig_main[0][0][85][0][0][9]
        #self.s_lambda = self.hfig_main[0][0][85][0][0][11]
        #self.A = self.hfig_main[0][0][85][0][0][10]

        #self.truncWinX =  self.hfig_main[0][0][85][0][0][7] #We really only need two numbers, not the whole list
        #self.truncWinY =  self.hfig_main[0][0][85][0][0][8]

        self.truncWinX =  self.hfig_main.calculation.truncWinX
        #We really only need two numbers, not the whole list
        self.truncWinY =  self.hfig_main.calculation.truncWinY

        #self.atomImage_trunc = self.atomImage[self.truncWinY[0,0]:self.truncWinY[0,-1],self.truncWinX[0,0]:self.truncWinX[0,-1]] #Do we really want to carry these around?
        #self.lightImage_trunc = self.lightImage[self.truncWinY[0,0]:self.truncWinY[0,-1],self.truncWinX[0,0]:self.truncWinX[0,-1]]
        #self.darkImage_trunc = self.darkImage[self.truncWinY[0,0]:self.truncWinY[0,-1],self.truncWinX[0,0]:self.truncWinX[0,-1]]
        self.atomImage_trunc = self.atomImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]] 
                                              #Do we really want to carry these around?
        self.lightImage_trunc = self.lightImage[self.truncWinY[0]:self.truncWinY[-1],
                                                self.truncWinX[0]:self.truncWinX[-1]]
        self.darkImage_trunc = self.darkImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]]
                                              
        self.flucWinX = self.hfig_main.calculation.flucWinX
        self.flucWinY = self.hfig_main.calculation.flucWinY
<<<<<<< HEAD
        intAtom = np.mean(np.mean(self.atomImage[self.flucWinY[0]:self.flucWinY[-1], self.flucWinX[0]:self.flucWinX[-1]]))
        intLight = np.mean(np.mean(self.lightImage[self.flucWinY[0]:self.flucWinY[-1], self.flucWinX[0]:self.flucWinX[-1]]))
        self.flucCor = intAtom / intLight
=======
        intAtom = np.mean(np.mean(self.atomImage[self.flucWinY[0]:self.flucWinY[-1],
                                              self.flucWinX[0]:self.flucWinX[-1]]))
        intLight = np.mean(np.mean(self.lightImage[self.flucWinY[0]:self.flucWinY[-1],
                                                self.flucWinX[0]:self.flucWinX[-1]]))
        self.flucCor = intAtom / intLight  
>>>>>>> added linear bias to fits

        
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
#        sub_block_data = self.runDataFiles[0,0][10]
#        sub_block_list = sub_block_data[:,0]
#        num_sub_blocks = sub_block_list.size
#        sub_blocks = {}
#        for i in np.arange(num_sub_blocks):
#            sub_blocks[sub_block_data[i,0][0]] = sub_block_data[i,1][0].astype(str)
#        return sub_blocks


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
        
    def getAtomNumber(self, axis=1, offset_switch = True, flucCor_switch = True, debug_flag = False, linear_bias_switch = True):
        ODImage = self.getODImage(flucCor_switch)
        imgcut = np.sum(ODImage,axis)
        try:
            if linear_bias_switch:
                coefs = self.fitGaussian1D(imgcut)
            else:
                coefs = self.fitGaussian1D_noline(imgcut)
        except:
            return 0
        offset = coefs[3]
        if offset_switch:
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
        
        coef, outmat = curve_fit(self.gaussian1D,xdata,image,p0 = p_0)
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
    def getPos(self, axis = 0, flucCor_switch = True):
        image = self.getODImage(flucCor_switch)
        imgcut = np.sum(image,axis)
        coefs = self.fitGaussian1D(imgcut)
        return coefs[1]*self.pixel_size
    def getWidth(self, axis=0):
        image = self.getODImage()
        imgcut = np.sum(image,axis)
        coefs = self.fitGaussian1D(imgcut)
        return coefs[2]*self.pixel_size
    def getLightCounts(self):
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
        
        
