'''code for loading condensate images and manipulating them'''
from cloud_distribution import CloudDistribution as CD

bec4 = CD(r'D:\ACMData\Condensate\2014-09-15\tof_4ms\\');
bec5 = CD(r'D:\ACMData\Condensate\2014-09-15\tof_5ms\\');
bec6 = CD(r'D:\ACMData\Condensate\2014-09-15\tof_6ms\\');
bec7 = CD(r'D:\ACMData\Condensate\2014-09-15\tof_7ms\\');
bec8= CD(r'D:\ACMData\Condensate\2014-09-15\tof_8ms\\');
bec9 = CD(r'D:\ACMData\Condensate\2014-09-15\tof_9ms\\');

avg4 = bec4.get_average_image();
avg5 = bec5.get_average_image();
avg6 = bec6.get_average_image();
avg7 = bec7.get_average_image();
avg8 = bec8.get_average_image();
avg9 = bec9.get_average_image();
