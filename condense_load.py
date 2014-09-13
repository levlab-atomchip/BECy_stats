'''code for loading condensate images and manipulating them'''
from cloud_distribution import CloudDistribution as CD

bec1 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_1ms\\')
bec3 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_3ms\\')
bec6 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_6ms\\')
bec9 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_9ms\\')
bec12 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_12ms\\')
bec20= CD(r'D:\ACMData\Condensate\2014-09-11\tof_20ms\\')
bec30 = CD(r'D:\ACMData\Condensate\2014-09-11\tof_30ms\\')

avg1 = bec1.get_average_image()
avg3 = bec3.get_average_image()
avg6 = bec6.get_average_image()
avg9 = bec9.get_average_image()
avg12 = bec12.get_average_image()
avg20 = bec20.get_average_image()
avg30 = bec30.get_average_image()