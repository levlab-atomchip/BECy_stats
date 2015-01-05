from cloud_image import CloudImage
import numpy as np

class NoAtomImage(CloudImage):
    def __init__(self, filename):
        CloudImage.__init__(self, filename)
        
    def get_cd_image(self
                    , axis=1
                    , linear_bias_switch=False
                    , INTCORR=True
                    , **kwargs
                    ):
        '''return the column density, with offset removed'''
        if INTCORR:
            od_image = self.optical_depth()
        else:
            od_image = self.get_od_image(**kwargs)
        imgcut = np.sum(od_image, axis)
        offset = np.mean(imgcut) / od_image.shape[1]

        return (od_image - offset) / self.s_lambda