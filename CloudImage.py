import scipy
import scipy.io
import parameters
import re
from scipy.optimize import curve_fit
from scipy.constants import pi, hbar
import numpy as np

class CloudImage():
    def __init__(self,fileName,p=None):
        if p==None:
            self.p = parameters.bfield_parameters()
        else:
            self.p=p
        
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
        # self.c1 (pixel size times magnification?)
        # self.s_lambda
        # self.A (area of pixel)

    def loadMatFile(self):
        scipy.io.loadmat(self.fileName,mdict = self.matFile, squeeze_me = True, struct_as_record = False)
        self.imageArray = self.matFile['rawImage']
        self.runDataFiles = self.matFile['runData']
        self.hfig_main = self.matFile['hfig_main']

        # print self.runDataFiles[0][0][13]
        
        self.ContParName = self.runDataFiles.ContParName
        # print(self.ContParName)
        self.CurrContPar = self.runDataFiles.CurrContPar
        self.CurrTOF = self.runDataFiles.CurrTOF*1e-3
        
        self.atomImage = scipy.array(self.imageArray[:,:,0]) 
        #scipy.array is called to make a copy, not a reference
        self.lightImage = scipy.array(self.imageArray[:,:,1])
        self.darkImage = scipy.array(self.imageArray[:,:,2])

        self.magnification = self.hfig_main.calculation.M
        self.pixel_size = self.hfig_main.calculation.pixSize
        self.imageRotation = self.hfig_main.display.imageRotation
        self.c1 = self.hfig_main.calculation.c1
        self.s_lambda = self.hfig_main.calculation.s_lambda
        self.A = self.hfig_main.calculation.A

        self.truncWinX =  self.hfig_main.calculation.truncWinX
        #We really only need two numbers, not the whole list
        self.truncWinY =  self.hfig_main.calculation.truncWinY
        
        self.flucWinX = self.hfig_main.calculation.flucWinX
        self.flucWinY = self.hfig_main.calculation.flucWinY
        intAtom = np.mean(np.mean(self.atomImage[self.flucWinY[0]:self.flucWinY[-1],
                                              self.flucWinX[0]:self.flucWinX[-1]]))
        intLight = np.mean(np.mean(self.lightImage[self.flucWinY[0]:self.flucWinY[-1],
                                                self.flucWinX[0]:self.flucWinX[-1]]))
        self.flucCor = intAtom / intLight
        
        self.atomImage_trunc = self.atomImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]] 
                                              #Do we really want to carry these around?
        self.lightImage_trunc = self.lightImage[self.truncWinY[0]:self.truncWinY[-1],
                                                self.truncWinX[0]:self.truncWinX[-1]]
        self.darkImage_trunc = self.darkImage[self.truncWinY[0]:self.truncWinY[-1],
                                              self.truncWinX[0]:self.truncWinX[-1]]
        return
    
    def truncate_image(self, x1, x2, y1, y2):
        self.atomImage_trunc = self.atomImage[y1:y2,x1:x2]
        self.lightImage_trunc = self.lightImage[y1:y2,x1:x2]
        self.darkImage_trunc = self.darkImage[y1:y2,x1:x2]


    def getVariablesFile(self):
        sub_block_data = self.runDataFiles[0][0][12]
        sub_block_list = sub_block_data[:,0]
        num_sub_blocks = sub_block_list.size
        sub_blocks = {}
        for i in np.arange(num_sub_blocks):
            sub_blocks[sub_block_data[i,0][0]] = sub_block_data[i,1][0].astype(str)
        return sub_blocks

    def getVariableValues(self,variable_file):
        sub_blocks = self.getVariablesFile()
        variables = sub_blocks[variable_file]
        variables = variables.replace(' ','').replace('\r','')
        variables = variables.split('\n')
        variables = [i.split('%')[0] for i in variables]
        variables = [i for i in variables if i != '']
        variables = [i.replace(';','') for i in variables]
        variables_dict = {}
        for i in variables:
            if i.find('=') > 0:
                var_name,var_val = i.split('=')
                variables_dict[var_name] = var_val
        return variables_dict

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return [array[idx],idx]

    def getODImage(self):
        ODImage = abs(np.log((self.atomImage_trunc 
                            - self.darkImage_trunc).astype(float)
                            /(self.flucCor * self.lightImage_trunc 
                            - self.darkImage_trunc).astype(float)))
        ODImage[np.isnan(ODImage)] = 0
        ODImage[np.isinf(ODImage)] = ODImage[~np.isinf(ODImage)].max()
        return ODImage
        
    def getAtomNumber(self, axis=1):
        ODImage = self.getODImage()
        imgcut = np.sum(ODImage,axis)
        coefs = self.fitGaussian1D(imgcut)
        offset = coefs[3]
        atomNumber = self.A/self.s_lambda*(np.sum(ODImage) - offset*len(imgcut));
        return atomNumber

    def gaussian1D(self,x,A,mu,sigma,offset):
        return A*np.exp(-1.*(x-mu)**2./(2.*sigma**2.)) + offset

    def gaussian2D(self,xdata, A_x,mu_x,sigma_x,A_y,mu_y,sigma_y,offset):
        x = xdata[0]
        y = xdata[1]
        return (A_x*np.exp(-1.*(x-mu_x)**2./(2.*sigma_x**2.)) 
        + A_y*np.exp(-1.*(y-mu_y)**2./(2.*sigma_y**2.)) + offset)
    
    def fitGaussian1D(self,image): #fits a 1D Gaussian to a 1D image
        max_value = image.max()
        max_loc = np.argmax(image)
        [half_max,half_max_ind] = self.find_nearest(image,max_value/2.)
        hwhm = abs(half_max_ind - max_loc)
        p_0 = [max_value,max_loc,hwhm,0.] #fit guess
        xdata = np.arange(np.size(image))
        
        try:
            coef, outmat = curve_fit(self.gaussian1D,xdata,image,p0 = p_0)
        except:
            coef = p_0
        return coef

    def fitGaussian2D(self,image): #fits a 2D Gaussian to a 2D Image
        img_x = np.sum(image,0)
        img_y = np.sum(image,1)
        x_coefs= self.fitGaussian1D(img_x) #gets coefficient estimates from 1D fits
        y_coefs= self.fitGaussian1D(img_y)
        x,y = np.meshgrid(np.arange(img_x.size),np.arange(img_y.size))

        coef, outmat = curve_fit(self.gaussian2D,[x,y],
                                 image,
                                 p0 =np.delete(np.append(x_coefs,y_coefs),3))
        return coef

    def getDensity1D(self,image):
        return self.A/self.s_lambda * image

    def getBField(self,m_F = 2):
        image_1D = np.sum(self.getODImage(),0)
        density = self.getDensity1D(image_1D)
        # This looks wrong, a sensitivity rather than a local B field
        return 2*hbar*self.p.w_tr*self.p.a_s/(m_F*self.p.g_F*self.p.mu_0)*density

    def getContParam(self):
        if self.ContParName == 'VOID':
            return 'void'
        else:
            Cont_Param = self.getVariableValues('Variables.m')[self.ContParName]
            return Cont_Param[self.CurrContPar]
    def getParamVal(self, param_name):
        return self.getVariableValues('Variables.m')[param_name]
    def getPos(self, axis = 0):
        image = self.getODImage()
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
    
        
        
