#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Will
#
# Created:     21/01/2014
# Copyright:   (c) Will 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cloud_distribution

def setup():
    directory = r'C:\Users\Will\Desktop\ODT_XFit_Test\quiktest\\'
    dist = cloud_distribution.CloudDistribution(directory)
    return dist