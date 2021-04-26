import h5py
import numpy as np
import sys
from matplotlib import pyplot as plt

hf = h5py.File('stats_30_days.h5', 'r')

total_bits = np.array(hf.get('total_bits')[:, :])
unique_bits = np.array(hf.get('unique_bits')[:, :])
hf.close()

static_unique = unique_bits[:, 0]
adaptive_unique = unique_bits[:, 1]
warm_static_unique = unique_bits[:, 2]
warm_adaptive_unique = unique_bits[:, 3]
rgct_unique = unique_bits[:, 4]
rrct_unique = unique_bits[:, 5]
uni_unqiue = unique_bits[:, 6]
cross_unique = unique_bits[:, 7]
pos_unqiue = unique_bits[:, 8]

total_mask = total_bits[:, 0]
static_mask = total_bits[:, 1]
adaptive_mask = total_bits[:, 2]
warm_static_mask = total_bits[:, 3]
warm_adaptive_mask = total_bits[:, 4]
rgct_mask = total_bits[:, 5]
rrct_mask = total_bits[:, 6]
uni_mask = total_bits[:, 7]
cross_mask = total_bits[:, 8]
pos_mask = total_bits[:, 9]

nb = 200
max_val = 1
array_bins = np.arange(0.0, max_val + max_val / nb, max_val / nb)

static_mask_norm = static_mask / total_mask
adaptive_mask_norm = adaptive_mask / total_mask
warm_static_mask_norm = warm_static_mask / total_mask
warm_adaptive_mask_norm = warm_adaptive_mask / total_mask

plt.hist(static_mask_norm, color="blue", bins=array_bins, label='Cold_Static')
plt.hist(adaptive_mask_norm, color="red", bins=array_bins, label='Cold_Adaptive')
plt.hist(warm_static_mask_norm, color="orange", bins=array_bins, label='Warm_Static')
plt.hist(warm_adaptive_mask_norm, color="green", bins=array_bins, label='Warm_Adaptive')
plt.legend()
plt.show()

rgct_mask_norm = rgct_mask / total_mask
uni_mask_norn = uni_mask / total_mask
cross_mask_norm = cross_mask / total_mask
pos_mask_norm = pos_mask / total_mask

plt.hist(rgct_mask_norm, color="blue", bins=array_bins, label='RGCT')
plt.hist(uni_mask_norn, color="red", bins=array_bins, label='Uniformity')
plt.hist(cross_mask_norm, color="orange", bins=array_bins, label='Cross Correlation')
plt.hist(pos_mask_norm, color="green", bins=array_bins, label='Positive Outliers')
plt.legend()
plt.show()

static_mask_uniq = static_unique / static_mask
adaptive_mask_uniq = adaptive_unique / adaptive_mask
warm_static_mask_uniq = warm_static_unique / warm_static_mask
warm_adaptive_mask_uniq = warm_adaptive_unique / warm_adaptive_mask


nb = 100
max_val = 1
array_bins = np.arange(0.0, max_val + max_val / nb, max_val / nb)



plt.hist(adaptive_mask_uniq, color="red", bins=array_bins, label='Cold_Adaptive')
plt.hist(static_mask_uniq, color="blue", bins=array_bins, label='Cold_Static')
plt.legend()
plt.show()



plt.hist(warm_static_mask_uniq, color="orange", bins=array_bins, label='Warm_Static')
plt.hist(warm_adaptive_mask_uniq, color="green", bins=array_bins, label='Warm_Adaptive')
plt.legend()
plt.show()




rgct_mask_uniq = rgct_unique / rgct_mask
uni_mask_uniq = uni_unqiue/ uni_mask
cross_mask_uniq = cross_unique / cross_mask
pos_mask_uniq = pos_unqiue / pos_mask


plt.hist(rgct_mask_uniq, color="blue", bins=array_bins, label='RGCT')
plt.hist(uni_mask_uniq, color="red", bins=array_bins, label='Uniformity')

plt.legend()
plt.show()



plt.hist(cross_mask_uniq, color="orange", bins=array_bins, label='Cross Correlation')
plt.hist(pos_mask_uniq, color="green", bins=array_bins, label='Positive Outliers')


plt.legend()
plt.show()


print(total_bits.shape)
sys.exit()

print(individual_bits.shape)
static_filter = individual_bits[:, 2]
adaptive_filter = individual_bits[:, 3] - static_filter
rgct_filter = individual_bits[:, 4]
cross_corr_filter = individual_bits[:, 7]
unifrom_filter = individual_bits[:, 6]
plt.hist(static_filter, color="blue", bins=range(0, 150000000, 150000), label='static_filter')
plt.hist(adaptive_filter, color="red", bins=range(0, 150000000, 150000), label='adaptive_filter')
plt.hist(rgct_filter, color="orange", bins=range(0, 150000000, 150000), label='rgct_filter')
plt.hist(unifrom_filter, color="green", bins=range(0, 150000000, 150000), label='unifrom_filter')
plt.hist(cross_corr_filter, color="magenta", bins=range(0, 150000000, 150000), label='cross_corr_filter')
plt.xlim([0, 15000000])
plt.legend()
plt.savefig("non_unique.png")
plt.show()
plt.close()

static_filter = unique_bits[:, 0]
adaptive_filter = unique_bits[:, 3]
rgct_filter = unique_bits[:, 4]
cross_corr_filter = unique_bits[:, 7]
unifrom_filter = unique_bits[:, 6]
plt.hist(static_filter, color="blue", bins=range(0, 150000000, 150000), label='static_filter')
plt.hist(adaptive_filter, color="red", bins=range(0, 150000000, 150000), label='adaptive_filter')
plt.hist(rgct_filter, color="orange", bins=range(0, 150000000, 150000), label='rgct_filter')
plt.hist(unifrom_filter, color="green", bins=range(0, 150000000, 150000), label='unifrom_filter')
plt.hist(cross_corr_filter, color="magenta", bins=range(0, 150000000, 150000), label='cross_corr_filter')
plt.xlim([0, 15000000])
plt.legend()
plt.savefig("unique.png")
plt.show()
