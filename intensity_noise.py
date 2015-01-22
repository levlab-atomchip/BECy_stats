from cloud_distribution_noatoms import CloudDistributionNoAtoms as CDna
from no_atom_image import NoAtomImage as nai
import matplotlib.pyplot as plt
import numpy as np

def image_subtract(im1, im2):
    return np.int32(im1) - np.int32(im2)

def get_intensity_noise(img, num_bins=50, threshold = 0):
    mean_img = image_subtract(0.5 * (img.atom_image_trunc + img.fluc_cor*img.light_image_trunc), img.dark_image_trunc)
    diff = image_subtract(img.fluc_cor*img.light_image_trunc, img.atom_image_trunc)

    mean_img_hist, mean_img_bins = np.histogram(np.ravel(mean_img), num_bins)

    diff_vec = np.ravel(diff)
    mean_vec = np.ravel(mean_img)
    diff_bin_labels = np.digitize(mean_vec, mean_img_bins)

    bin_members = [[] for _ in range(len(mean_img_bins))]
    for ii, this_diff in enumerate(diff_vec):
        bin_members[diff_bin_labels[ii]-1].append(this_diff)
    bin_vars = [np.std(members)**2 for members in bin_members]

    threshold_counts = threshold * np.size(mean_img) / num_bins
    filtered_bin_vars = [0 if meancounts < threshold else bin_var for meancounts, bin_var in zip(mean_img_hist, bin_vars)]
    return mean_img_bins[:-1], filtered_bin_vars

if __name__ == "__main__":
    datadir = r'Z:\Data\ACM Data\Imaging system\PIXIS\2015-01-21\100us'
    dist = CDna(datadir, False)
    most_counts = 0
    sat_counts = 3400
    for ii, fname in enumerate(dist.filelist):
        print 'Processing File %d'%(ii+1)
        this_img = nai(fname)
        this_img.truncate_image(396, 623, 130, 148)
        this_bins, this_vars = get_intensity_noise(this_img)
        this_sat = this_bins 
        plt.plot(this_sat, this_vars, '.', color='b')
        most_counts = max((most_counts, max(this_bins)))
    plt.plot([0, float(most_counts)], [0, 2.0*most_counts], color='red', linewidth=3, label='Shot Noise Limited')
   # plt.xlim(0, 2)
   # plt.ylim(0, 30000)
    plt.xlabel('Photon Counts')
    plt.ylabel('Variance')
    plt.legend(loc='upper left')
    plt.show(block=True)
