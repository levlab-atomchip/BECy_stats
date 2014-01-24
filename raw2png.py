# script for converting imaging gui rawimage files to pngs rapidly

import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def raw2png(directory, VIEW=1, ROI=1):

    filelist = glob.glob(directory + '*.mat')

    for file in filelist:
        mat_dict = {}
        try:
            scipy.io.loadmat(file, mdict=mat_dict,
                            squeeze_me=True, struct_as_record=False)
        except:
            print(file + ' failed')
            pass
        hfig_main = mat_dict['hfig_main']
        rawimages = mat_dict['rawImage']
        atom_image = np.nan_to_num(rawimages[:,:,0])
        light_image = np.nan_to_num(rawimages[:,:,1])
        dark_image = np.nan_to_num(rawimages[:,:,2])
        if ROI:
            trunc_win_x = hfig_main.calculation.truncWinX
            trunc_win_y = hfig_main.calculation.truncWinY

            atom_image = \
            atom_image[trunc_win_y[0]:trunc_win_y[-1],
                            trunc_win_x[0]:trunc_win_x[-1]]
            light_image = \
            light_image[trunc_win_y[0]:trunc_win_y[-1],
                            trunc_win_x[0]:trunc_win_x[-1]]
            dark_image = \
            dark_image[trunc_win_y[0]:trunc_win_y[-1],
                            trunc_win_x[0]:trunc_win_x[-1]]

        ODImg = np.absolute(np.log(atom_image + 1) - np.log(light_image + 1))
        print(file + ' processed')
        plt.imshow(ODImg, cmap = 'jet')
        plt.title(file)
        if VIEW:
            plt.show()
        else:
            plt.savefig(file[:-4]+'.png')
    print('Directory %s Processed'%directory)

if __name__ == "__main__":
    directory = r'D:\ACMData\Statistics\mac_capture_number\2014-01-23\\'
    raw2png(directory)
