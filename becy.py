#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      LevLab
#
# Created:     27/01/2014
# Copyright:   (c) LevLab 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def setup():
    import cloud_distribution
    directory = r'D:\ACMData\Stability\moveDipole_vary_posend\2014-01-27\\'
    dist = cloud_distribution.CloudDistribution(directory)
    return dist

def main():
    pass
if __name__ == '__main__':
    main()
