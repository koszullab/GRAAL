__author__ = 'hervemn'
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
from scipy.stats import norm

class data():
    def __init__(self, folder, output_folder):
        self.folder = folder
        self.list_data = ['scale_factor', 'slope', 'val_trans', 'dist_max_intra', 'n_contigs', 'mean_length_contigs',
                       'distance_init_genome', 'likelihood']
        self.dict_data = dict()
        self.dict_data['scale_factor'] = dict()
        self.dict_data['slope'] = dict()
        self.dict_data['val_trans'] = dict()
        self.dict_data['dist_max_intra'] = dict()
        self.dict_data['n_contigs'] = dict()
        self.dict_data['mean_length_contigs'] = dict()
        self.dict_data['distance_init_genome'] = dict()
        self.dict_data['likelihood'] = dict()

        self.dict_data['scale_factor']['txt_file'] = os.path.join(self.folder, 'list_fact.txt')
        self.dict_data['slope']['txt_file'] = os.path.join(self.folder, 'list_slope.txt')
        self.dict_data['val_trans']['txt_file'] = os.path.join(self.folder, 'list_d_nuc.txt')
        self.dict_data['dist_max_intra']['txt_file'] = os.path.join(self.folder, 'list_d_max.txt')
        self.dict_data['n_contigs']['txt_file'] = os.path.join(self.folder, 'list_n_contigs.txt')
        self.dict_data['mean_length_contigs']['txt_file'] = os.path.join(self.folder, 'list_mean_len.txt')
        self.dict_data['distance_init_genome']['txt_file'] = os.path.join(self.folder, 'list_dist_init_genome.txt')
        self.dict_data['likelihood']['txt_file'] = os.path.join(self.folder, 'list_likelihood.txt')

        self.n = len(self.list_data)
        for i in range(0, self.n):
            key = self.list_data[i]
            f = self.dict_data[key]['txt_file']
            h = open(f, 'r')
            all_lines = h.readlines()
            data = [float(ele) for ele in all_lines]
            self.dict_data[key]['data'] = data
            h.close()



    def make_multi_plot(self, lim_burn_in):
        plt.figure(figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(4, 3)
        gs.update(wspace=0.1, hspace=0.5)
        size_label = 7
        size_label_y = 10
        size_title = 12
        size_tick  = 5
        for i in range(0, 4):
            key = self.list_data[i]
            data = self.dict_data[key]['data']
            ax1 = plt.subplot(gs[i, :-1])
            ax2 = plt.subplot(gs[i,-1])

            ax1.plot(data, 'k')
            ax1.set_xlabel('iteration',fontsize=size_label)
            ax1.set_ylabel(key, fontsize=size_label_y)
            if i == 0:
                # ax1.set_title('Trace ' + key,fontsize=size_title)
                ax1.set_title('Trace',fontsize=size_title)
            plt.setp(ax1.get_xticklabels(), fontsize=size_tick)
            plt.setp(ax1.get_yticklabels(), fontsize=size_tick)
            ax1.axvline(lim_burn_in, linestyle='dashed', color='g', linewidth=1)

            n, bins, patches = ax2.hist(data[lim_burn_in:], 30, normed=1, facecolor='blue', alpha=0.75)
            # if i == 0:
                # ax2.set_title('Distribution of '+ key, fontsize=size_title)
                # ax2.set_title('Distribution', fontsize=size_title)

            ax2.set_xlabel(key,fontsize=size_label)
            plt.setp(ax2.get_xticklabels(), fontsize=size_tick)
            plt.setp(ax2.get_yticklabels(), fontsize=size_tick)

            #    add a 'best fit' line
            (mu, sigma) = norm.fit(data[lim_burn_in:])
            y = mlab.normpdf( bins, mu, sigma)
            l = ax2.plot(bins, y, 'r-', linewidth=2)
            ax2.set_title(r'$\mu=%.3f,\ \sigma=%.5f$' %(mu, sigma))

        plt.show()