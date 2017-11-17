# coding: utf-8
__author__ = 'hervemn'
import numpy as np
import pycuda
import pycuda.tools

from pycuda import characterize

from pycuda import *

import pycuda.characterize

import pycuda.driver as cuda
import pycuda.compiler
from gpustruct import GPUStruct
from pycuda import gpuarray as ga
import pycuda.gl as cudagl
import time


import matplotlib.pyplot as plt
import optim_rippe_curve_update as opti
import optim_hic_curve as opti_rv
from OpenGL.arrays import vbo
from scipy import stats
import timing
from PIL import Image
timings = timing.Timing()



class sampler():
    def __init__(self, use_rippe, S_o_A_frags, collector_id_repeats, frag_dispatcher,
                 id_frag_duplicated, id_frags_blacklisted,
                 n_frags, n_new_frags, init_n_sub_frags, n_new_sub_frags, np_rep_sub_frags_id,
                 hic_matrix_sub_sampled,
                 np_sub_frags_len_bp, np_sub_frags_id, np_sub_frags_accu,
                 mean_squared_frags_per_bin, norm_vect_accu,
                 S_o_A_sub_frags,
                 hic_matrix, mean_value_trans, n_iterations, is_simu, gl_window, pos_vbo, col_vbo, vel, pos,
                 raw_im_init,pbo_im_buffer,
                 sub_sample_factor):
        self.o = 0
        self.sub_sample_factor = sub_sample_factor
        if self.sub_sample_factor <= 1 and self.sub_sample_factor > 0:
            self.perform_sub_sample = True
        else:
            self.perform_sub_sample = False

        self.raw_im_init = raw_im_init
        self.pbo_im_buffer = pbo_im_buffer

        self.use_rippe = use_rippe
        self.gl_window = gl_window
        self.ctx = gl_window.ctx_gl
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo
        self.pos = pos
        self.vel = vel
        self.load_gl_cuda_vbo()

        self.id_frags_blacklisted = id_frags_blacklisted
        self.int4 = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32),('n', np.int32)], align=True)
        self.float3 = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)], align=True)
        self.int3 = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32)], align=True)

        self.np_id_frag_duplicated = np.int32(id_frag_duplicated)
        self.id_frag_duplicated = id_frag_duplicated
        self.n_frags = np.int32(n_frags)
        self.n_new_frags = np.int32(n_new_frags)
        self.init_n_sub_frags = np.int32(init_n_sub_frags)
        self.n_new_sub_frags = np.int32(n_new_sub_frags)

        self.uniq_frags = np.int32(np.lib.arraysetops.setdiff1d(np.arange(0, self.n_frags, dtype=np.int32),
                                                                self.np_id_frag_duplicated))
        self.n_frags_uniq = np.int32(len(self.uniq_frags))

        self.n_values_triu = np.int32(self.n_frags * (self.n_frags - 1) / 2)
        self.init_n_values_triu = np.int32(self.n_frags * (self.n_frags - 1) / 2)
        self.new_n_values_triu = np.int32(self.n_new_frags * (self.n_new_frags - 1) / 2)
        self.init_n_sub_values_triu = np.int32(self.init_n_sub_frags * ( self.init_n_sub_frags - 1) / 2)
        self.new_n_sub_values_triu = np.int32(self.n_new_sub_frags * ( self.n_new_sub_frags - 1) / 2)

        self.init_n_values_triu_extra = self.init_n_values_triu + self.n_frags
        self.init_n_sub_values_triu_extra = self.init_n_sub_values_triu + self.init_n_sub_frags



        self.cpu_likelihood = np.zeros((self.init_n_values_triu_extra,), dtype=np.float64)
        self.gpu_likelihood = ga.to_gpu(ary=self.cpu_likelihood)

        self.curr_likelihood = ga.to_gpu(np.zeros((self.init_n_values_triu_extra,), dtype=np.float64))
        self.curr_likelihood_forward = ga.to_gpu(np.zeros((self.init_n_values_triu_extra,), dtype=np.float64))


        # self.cpu_full_expected_lin = np.zeros((self.init_n_sub_values_triu, ), dtype=np.float32)
        # self.gpu_full_expected_lin = ga.to_gpu(ary=self.cpu_full_expected_lin)
        self.cpu_full_likelihood = np.zeros((self.init_n_sub_values_triu,), dtype=np.float64)
        self.gpu_full_likelihood = ga.to_gpu(ary=self.cpu_full_likelihood)

        # self.cpu_2d_full_expected_mat = np.zeros((self.init_n_sub_frags, self.init_n_sub_frags), dtype=np.float32)
        # self.gpu_2d_full_expected_mat = ga.to_gpu(ary=self.cpu_2d_full_expected_mat)

        # self.cpu_full_expected_rep_lin = np.zeros((self.new_n_sub_values_triu, ), dtype=np.float32)
        # self.gpu_full_expected_rep_lin = ga.to_gpu(ary=self.cpu_full_expected_rep_lin)

        # self.cpu_2d_full_expected_rep_mat = np.zeros((self.n_new_sub_frags, self.n_new_sub_frags), dtype=np.float32)
        # self.gpu_2d_full_expected_rep_mat = ga.to_gpu(ary=self.cpu_2d_full_expected_rep_mat)


        self.n_modif_metropolis = 13
        self.n_tmp_struct = 13
        self.is_simu = is_simu
        self.norm_vect_accu = norm_vect_accu
        self.np_sub_frags_len_bp = np_sub_frags_len_bp
        self.np_sub_frags_id = np_sub_frags_id
        self.np_sub_frags_accu = np_sub_frags_accu
        self.hic_matrix_sub_sampled = hic_matrix_sub_sampled
        self.mean_squared_frags_per_bin = np.float32(mean_squared_frags_per_bin)
        print "size hic matrix = ", hic_matrix.nbytes/10**6
        ################################################################################################################
        self.collector_id_repeats = collector_id_repeats
        self.gpu_collector_id_repeats = ga.to_gpu(ary=self.collector_id_repeats)
        self.frag_dispatcher = frag_dispatcher
        self.gpu_frag_dispatcher = cuda.mem_alloc(self.frag_dispatcher.nbytes)
        cuda.memcpy_htod(self.gpu_frag_dispatcher, self.frag_dispatcher)
        self.np_rep_sub_frags_id = np_rep_sub_frags_id
        self.gpu_rep_sub_frags_id = cuda.mem_alloc_like(self.np_rep_sub_frags_id)
        cuda.memcpy_htod(self.gpu_rep_sub_frags_id, self.np_rep_sub_frags_id)
        self.gpu_uniq_frags = ga.to_gpu(ary=self.uniq_frags)

        ################################################################################################################
        self.S_o_A_frags = S_o_A_frags
        self.S_o_A_sub_frags = S_o_A_sub_frags
        self.mean_value_trans = mean_value_trans
        self.param_simu_rippe = np.dtype([('kuhn', np.float32), ('lm', np.float32), ('c1', np.float32), ('slope', np.float32),
                                      ('d', np.float32), ('l_max', np.float32), ('fact', np.float32),
                                      ('v_inter', np.float32)], align=True)
        self.param_simu_exp = np.dtype([('d0', np.float32), ('d1', np.float32), ('d_max', np.float32),
                                      ('alpha_0', np.float32), ('alpha_1', np.float32), ('alpha_2', np.float32),
                                      ('fact', np.float32),
                                      ('v_inter', np.float32)], align=True)
        if self.use_rippe:
            self.param_simu_T = self.param_simu_rippe
        else:
            self.param_simu_T = self.param_simu_exp
        ################################################################################################################
        self.n_iterations = n_iterations
        (free, total)=cuda.mem_get_info()
        print("Global memory occupancy before init:%f%% free"%(free*100/total))
        print("Global free memory before init:%i Mo free"%(free/10**6))
        ################################################################################################################
        self.sub_width_mat = np.int32(hic_matrix.shape[0])
        self.sub_n_frags = np.int32(self.sub_width_mat)
        self.hic_matrix_ori = np.copy(np.float32(hic_matrix))
        self.hic_matrix = np.copy(np.float32(hic_matrix))
        idx_diag = np.diag_indices_from(self.hic_matrix)
        self.hic_matrix[idx_diag] = 0
        idx_diag_sub = np.diag_indices_from(self.hic_matrix_sub_sampled)
        self.hic_matrix_sub_sampled[idx_diag_sub] = 0
        for id_f in self.id_frags_blacklisted:
            real_id = self.S_o_A_frags['id_d'][id_f]
            self.hic_matrix_sub_sampled[real_id, :] = 0
            self.hic_matrix_sub_sampled[:, real_id] = 0
            da = self.np_sub_frags_id[real_id]

            for i in range(0, da['w']):
                id = da[i]
                self.hic_matrix[id,:] = self.mean_value_trans
                self.hic_matrix[:, id] =self.mean_value_trans
                self.hic_matrix_ori[id,:] = self.mean_value_trans
                self.hic_matrix_ori[:, id] = self.mean_value_trans

        ############### DEBUGGG #############################
        #da1 = self.np_sub_frags_id[71]
        #
        #da2 = self.np_sub_frags_id[72]
        #da3 = self.np_sub_frags_id[79]
        #
        #self.hic_matrix[da1['x'], da2['x']:da3['z']] = self.mean_value_trans
        #self.hic_matrix[da2['x']:da3['z'], da1['x']] = self.mean_value_trans
        #
        #self.hic_matrix_sub_sampled[71, 72:79] = 0
        #self.hic_matrix_sub_sampled[72:79, 71] = 0

        ############### DEBUGGG #############################

        self.define_neighbourhood()

        self.n_new_values_triu = np.int32(self.n_new_frags * (self.n_new_frags - 1) / 2)
        self.sub_n_values_triu = np.int32(self.sub_n_frags * (self.sub_n_frags - 1) / 2)
        ################################################################################################################

        self.gpu_2d_hic_matrix = ga.to_gpu(self.hic_matrix)
        self.cpu_simu_matrix = np.zeros((self.hic_matrix.shape), dtype=np.float32)
        self.cpu_simu_sub_matrix = np.zeros((self.hic_matrix_sub_sampled.shape), dtype=np.float32)
        self.gpu_2d_simu_hic_matrix = ga.to_gpu(self.cpu_simu_matrix)
        # if self.is_simu:
        # self.gpu_2d_sub_hic_matrix = ga.to_gpu(self.hic_matrix_sub_sampled)
        # self.cpu_2d_full_sub_hic_matrix = np.zeros((self.n_new_frags, self.n_new_frags), dtype=np.float32)
        # self.gpu_2d_full_sub_hic_matrix = ga.to_gpu(ary=self.cpu_2d_full_sub_hic_matrix)
        # self.cpu_full_mat_rep = np.zeros((self.n_new_sub_frags, self.n_new_sub_frags), dtype=np.float32)
        #
        # self.lin_full_mat_rep = np.zeros((self.n_sub_values_triu, ), dtype=np.float32)
        # self.cpu_full_mat_rep = np.zeros((self.n_new_sub_frags, self.n_new_sub_frags), dtype=np.float32)
        # self.gpu_full_mat_rep = ga.to_gpu(ary=self.lin_full_mat_rep)
        # self.gpu_2d_full_mat_rep = ga.to_gpu(ary=self.cpu_full_mat_rep)

        ################################################################################################################
        self.gpu_sub_frag_len_bp = cuda.mem_alloc(self.np_sub_frags_len_bp.nbytes)
        cuda.memcpy_htod(self.gpu_sub_frag_len_bp, self.np_sub_frags_len_bp)
        self.gpu_sub_frag_id = cuda.mem_alloc(self.np_sub_frags_id.nbytes)
        cuda.memcpy_htod(self.gpu_sub_frag_id, self.np_sub_frags_id)
        self.gpu_sub_frag_accu = cuda.mem_alloc(self.np_sub_frags_accu.nbytes)
        cuda.memcpy_htod(self.gpu_sub_frag_accu, self.np_sub_frags_accu)
        ################################################################################################################
        self.cpu_likelihood_fi = np.zeros((self.n_values_triu,), dtype=np.float64)
        print "size likelihood = ", self.cpu_likelihood_fi.nbytes / 10**6.
        self.gpu_sub_index = ga.to_gpu(np.zeros((self.n_new_frags,), dtype=np.int32))
        self.list_likelihood = []
        self.gpu_expected_mat = ga.to_gpu(np.zeros((self.n_values_triu,), dtype=np.float32))
        self.gpu_full_expected_mat = ga.to_gpu(np.zeros((self.n_values_triu,), dtype=np.float32))
        for i in range(0, self.n_tmp_struct):
            self.list_likelihood.append(ga.to_gpu(np.zeros((self.n_values_triu,), dtype=np.float32)))
        ########################
        self.np_init_prev = np.copy(self.S_o_A_frags['prev'])
        self.np_init_next = np.copy(self.S_o_A_frags['next'])
        self.np_init_orientable = []
        for idf in xrange(0, self.n_new_frags):
            id_d = self.S_o_A_frags['id_d'][idf]
            self.np_init_orientable.append(self.np_sub_frags_id[id_d]['w'] > 1)
        self.np_init_orientable = np.array(self.np_init_orientable, dtype=np.int32)
        self.np_init_ori = np.ones((self.n_new_frags,), dtype=np.int32)
        self.gpu_vect_frags_forward = GPUStruct([(np.int32, '*pos', np.copy(self.S_o_A_frags['pos'])),
                                                 (np.int32, '*id_c', np.copy(self.S_o_A_frags['id_c'])),
                                                 (np.int32, '*start_bp', np.copy(self.S_o_A_frags['start_bp'])),
                                                 (np.int32, '*len_bp', np.copy(self.S_o_A_frags['len_bp'])),
                                                 (np.int32, '*circ', np.copy(self.S_o_A_frags['circ'])),
                                                 (np.int32, '*id', np.copy(self.S_o_A_frags['id'])),
                                                 (np.int32, '*prev', np.copy(self.S_o_A_frags['prev'])),
                                                 (np.int32, '*next', np.copy(self.S_o_A_frags['next'])),
                                                 (np.int32, '*l_cont', np.copy(self.S_o_A_frags['l_cont'])),
                                                 (np.int32, '*l_cont_bp', np.copy(self.S_o_A_frags['l_cont_bp'])),
                                                 (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                 (np.int32, '*rep', np.copy(self.S_o_A_frags['rep'])),
                                                 (np.int32, '*activ', np.copy(self.S_o_A_frags['activ'])),
                                                 (np.int32, '*id_d', np.copy(self.S_o_A_frags['id_d']))])

        self.gpu_vect_frags = GPUStruct([(np.int32, '*pos', np.copy(self.S_o_A_frags['pos'])),
                                         (np.int32, '*id_c', np.copy(self.S_o_A_frags['id_c'])),
                                         (np.int32, '*start_bp', np.copy(self.S_o_A_frags['start_bp'])),
                                         (np.int32, '*len_bp', np.copy(self.S_o_A_frags['len_bp'])),
                                         (np.int32, '*circ', np.copy(self.S_o_A_frags['circ'])),
                                         (np.int32, '*id', np.copy(self.S_o_A_frags['id'])),
                                         (np.int32, '*prev', np.copy(self.S_o_A_frags['prev'])),
                                         (np.int32, '*next', np.copy(self.S_o_A_frags['next'])),
                                         (np.int32, '*l_cont', np.copy(self.S_o_A_frags['l_cont'])),
                                         (np.int32, '*l_cont_bp', np.copy(self.S_o_A_frags['l_cont_bp'])),
                                         (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                         (np.int32, '*rep', np.copy(self.S_o_A_frags['rep'])),
                                         (np.int32, '*activ', np.copy(self.S_o_A_frags['activ'])),
                                         (np.int32, '*id_d', np.copy(self.S_o_A_frags['id_d']))])

        self.gpu_vect_frags_forward.copy_to_gpu()
        self.gpu_vect_frags.copy_to_gpu()

        self.cpu_id_contigs = np.copy(self.S_o_A_frags['id_c'])
        self.gpu_id_contigs = ga.to_gpu(self.cpu_id_contigs)
        self.collector_gpu_vect_frags = []
        for k in xrange(0, max(self.n_tmp_struct, self.n_modif_metropolis)):
            self.collector_gpu_vect_frags.append(GPUStruct([(np.int32, '*pos', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*id_c', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*start_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*len_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*circ', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*id', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*prev', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*next', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*l_cont', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*l_cont_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*rep', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*activ', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                            (np.int32, '*id_d', np.zeros((self.n_new_frags,), dtype=np.int32))]))

        for k in xrange(0, max(self.n_tmp_struct, self.n_modif_metropolis)):
            self.collector_gpu_vect_frags[k].copy_to_gpu()

        self.pop_gpu_vect_frags = GPUStruct([(np.int32, '*pos', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id_c', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*start_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*len_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*circ', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*prev', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*next', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*l_cont', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*l_cont_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*rep', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*activ', np.ones((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id_d', np.zeros((self.n_new_frags,), dtype=np.int32))])

        self.scrambled_gpu_vect_frags = GPUStruct([(np.int32, '*pos', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id_c', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*start_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*len_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*circ', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*prev', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*next', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*l_cont', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*l_cont_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*rep', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*activ', np.ones((self.n_new_frags,), dtype=np.int32)),
                                            (np.int32, '*id_d', np.zeros((self.n_new_frags,), dtype=np.int32))])

        self.pop_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.pop_gpu_id_contigs = ga.to_gpu(self.pop_cpu_id_contigs)
        self.pop_gpu_vect_frags.copy_to_gpu()
        self.scrambled_gpu_vect_frags.copy_to_gpu()

        self.trans1_gpu_vect_frags = GPUStruct([(np.int32, '*pos', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id_c', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*start_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*len_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*circ', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*prev', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*next', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*l_cont', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*l_cont_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*rep', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*activ', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id_d', np.zeros((self.n_new_frags,), dtype=np.int32))])

        self.trans1_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.trans1_gpu_id_contigs = ga.to_gpu(self.trans1_cpu_id_contigs)
        self.trans1_gpu_vect_frags.copy_to_gpu()

        self.trans2_gpu_vect_frags = GPUStruct([(np.int32, '*pos', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id_c', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*start_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*len_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*circ', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*prev', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*next', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*l_cont', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*l_cont_bp', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*ori', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*rep', np.zeros((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*activ', np.ones((self.n_new_frags,), dtype=np.int32)),
                                                (np.int32, '*id_d', np.zeros((self.n_new_frags,), dtype=np.int32))])

        self.trans2_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.trans2_gpu_id_contigs = ga.to_gpu(self.trans2_cpu_id_contigs)
        self.trans2_gpu_vect_frags.copy_to_gpu()

        ########################
        self.n_generators = 2000
        seed = 1
        self.rng_states = cuda.mem_alloc(self.n_generators*pycuda.characterize.sizeof('curandStateXORWOW',
            '#include <curand_kernel.h>'))
        ################################################################################################################
        (free,total)=cuda.mem_get_info()
        print("Global memory occupancy after init:%f%% free"%(free*100./total))
        print("Global free memory after init:%i Mo free"%(free/10**6.))
        ################################################################################################################
        print "loading kernels ..."
        if self.use_rippe:
            self.loadProgram('kernels3.cu')
        else:
            self.loadProgram('kernels4.cu')
        print "kernels compiled"
        self.init_rng = self.module.get_function('init_rng')
        self.gen_rand_mat = self.module.get_function('gen_rand_mat')
        self.simulate_data_2d = self.module.get_function('simulate_data_2d')
        self.evaluate_likelihood = self.module.get_function('evaluate_likelihood')
        self.fill_2d_mat = self.module.get_function('fill_2d_contacts')
        self.fill_sub_index_A = self.module.get_function("fill_sub_index_fA")
        self.fill_sub_index_B = self.module.get_function("fill_sub_index_fB")
        self.sub_compute_likelihood_1d = self.module.get_function("sub_compute_likelihood")

        self.set_null = self.module.get_function("set_null")
        self.copy_gpu_array = self.module.get_function("copy_gpu_array")
        self.gl_update_pos = self.module.get_function("gl_update_pos")
        self.gl_update_im = self.module.get_function("reorder_tex")
        self.gpu_transloc = []
        self.pop_out = self.module.get_function('pop_out_frag')
        self.flip_frag = self.module.get_function('flip_frag')
        self.pop_in_1 = self.module.get_function('pop_in_frag_1') # split insert @ left
        self.pop_in_2 = self.module.get_function('pop_in_frag_2') # split insert @ right
        self.pop_in_3 = self.module.get_function('pop_in_frag_3') # insert @ left
        self.pop_in_4 = self.module.get_function('pop_in_frag_4') # insert @ right
        self.split = self.module.get_function('split_contig')
        self.paste = self.module.get_function('paste_contigs')
        self.simple_copy = self.module.get_function('simple_copy')
        self.copy_vect = self.module.get_function('copy_struct')
        self.swap_activity = self.module.get_function('swap_activity_frag')
        self.modification_str = ['eject frag',
                                 'flip frag',
                                 'pop out split insert @ left or 1', 'pop out split insert @ left or -1',
                                 'pop out split insert @ right or 1', 'pop out split insert @ right or -1',
                                 'pop out insert @ right or 1', 'pop out insert @ right or -1',
                                 'swap activity',
                                 # 'pop out insert @ left or 1', 'pop out insert @ left or -1',
                                 'transloc_1','transloc_2','transloc_3','transloc_4',
                                 'local_scramble d1', 'local_scramble d2', 'local_scramble d3', 'local_scramble d4']
        ################################################################################################################
        self.texref = self.module.get_texref("tex")
        self.load_gl_cuda_tex_buffer(self.raw_im_init)
        ################################################################################################################
        self.stride = 50
        self.index_4_metropolis_operations = []
        self.index_4_metropolis_operations.append(0)
        self.index_4_metropolis_operations.append(1)
        self.index_4_metropolis_operations.append(4)
        self.index_4_metropolis_operations.append(5)
        self.index_4_metropolis_operations.append(6)
        self.index_4_metropolis_operations.append(7)

        ################################################################################################################
        self.alpha_space = np.arange(-2, -0.4, .1)
        d_space_0 = np.arange(0.1, 2.1, 0.1)
        d_space_1 = np.arange(2, 100, 3)
        d_space = []
        d_space.extend(list(d_space_0))
        d_space.extend(list(d_space_1))
        self.d_space = np.array(d_space, np.float32)
        self.gpu_alpha_space = ga.to_gpu(self.alpha_space)
        self.gpu_d_space = ga.to_gpu(self.d_space)
        self.gpu_likeli_d_space = ga.to_gpu(ary=np.zeros(self.d_space.shape, dtype=np.float64))
        self.gpu_likeli_alpha_space = ga.to_gpu(ary=np.zeros(self.alpha_space.shape, dtype=np.float64))
        self.n_values_d_space = np.int32(len(self.d_space))
        self.n_values_alpha_space = np.int32(len(self.alpha_space))
        ################################################################################################################
        seed = 1
        self.init_rng(np.int32(self.n_generators), self.rng_states, np.uint64(seed), np.uint64(0), block=(64, 1, 1),
                      grid=(self.n_generators//64 + 1, 1))
        ################################################################################################################
        self.n_neighbors = 10
        self.setup_distri_frags()
        self.define_repeats()

    def init_likelihood(self):
        # self.likelihood_t = self.compute_likelihood(self.gpu_vect_frags, self.curr_likelihood)
        self.likelihood_t = self.eval_likelihood()

    def define_repeats(self):
        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        tmp_repeated = c.id != c.id_d
        id_repeat = np.unique(c.id_d[tmp_repeated])
        self.is_repeat = []
        self.n_frags_duplicated = 0
        tmp = []
        tmp.extend(self.id_frags_blacklisted)
        print "id_repeat = ", id_repeat
        for id_f in xrange(0, self.n_new_frags):
            if c.id_d[id_f] in id_repeat:
                self.is_repeat.append(True)
                self.n_frags_duplicated += 1
                tmp.append(id_f)
            else:
                self.is_repeat.append(False)
        self.n_frags_4_dist = len(np.unique(tmp))
        # print "n tot frags = ", self.n_new_frags
        # print "frags to remove dist = ", self.n_frags_4_dist
        # print "tmp = ", tmp
        # print "n frags duplicated = ", self.n_frags_duplicated

    def dist_inter_genome(self, tmp_gpu_vect_frags):
        tmp_gpu_vect_frags.copy_from_gpu()
        g1 = tmp_gpu_vect_frags
        n_frags_blacklisted = len(self.id_frags_blacklisted)
        # d = 3.0 * (self.n_new_frags - n_frags_blacklisted - self.n_frags_duplicated)
        d = 3.0 * (self.n_new_frags - self.n_frags_4_dist)
        # norm_distance = 3.0 * (self.n_new_frags - n_frags_blacklisted - self.n_frags_duplicated)
        norm_distance = 3.0 * (self.n_new_frags - self.n_frags_4_dist)
        # d = 3.0 * self.n_new_frags
        for id_f in xrange(0, self.n_new_frags):
            if id_f not in self.id_frags_blacklisted and not self.is_repeat[id_f]:
                prev_t0 = self.np_init_prev[id_f]

                tmp_prev_t1 = g1.prev[id_f]
                if tmp_prev_t1 != -1:
                    prev_t1 = g1.id_d[tmp_prev_t1]
                else:
                    prev_t1 = tmp_prev_t1

                next_t0 = self.np_init_next[id_f]

                tmp_next_t1 = g1.next[id_f]
                if tmp_next_t1 != -1:
                    next_t1 = g1.id_d[tmp_next_t1]
                else:
                    next_t1 = tmp_next_t1
                ori_t0 = self.np_init_ori[id_f]
                ori_t1 = g1.ori[id_f]
                swap = 1
                if ((prev_t1 == prev_t0) and (next_t1 == next_t0)) or ((prev_t1 == next_t0) and (next_t1 == prev_t0)):
                    d -= 1
                    # if not(self.np_init_orientable[id_f]):
                    #     d -= 2
                if self.np_init_orientable[id_f]:
                    if ori_t0 != ori_t1:
                        tmp = prev_t1
                        prev_t1 = next_t1
                        next_t1 = tmp
                        swap = -1
                    if prev_t0 == prev_t1:
                        if prev_t0 == -1:
                            d -= 1
                        elif not(self.np_init_orientable[prev_t1]):
                            d -= 1
                        else:
                            d -= 0.5
                            ori_prev_t0 = self.np_init_ori[prev_t0]
                            ori_prev_t1 = g1.ori[prev_t1]
                            if ori_prev_t0 == swap * ori_prev_t1:
                                d -= 0.5
                    if next_t0 == next_t1:
                        if next_t0 == -1:
                            d -= 1
                        elif not(self.np_init_orientable[next_t1]):
                            d -= 1
                        else:
                            d -= 0.5
                            ori_next_t0 = self.np_init_ori[next_t0]
                            ori_next_t1 = g1.ori[next_t1]
                            if ori_next_t0 == swap * ori_next_t1:
                                d -= 0.5
                else:
                    if ((prev_t1 == prev_t0) or (prev_t1 == next_t0)):
                        d -= 1
                    if ((next_t1 == next_t0) or (next_t1 == prev_t0)):
                        d -= 1
        return d / norm_distance

    def eval_likelihood(self,):
        start = cuda.Event()
        end = cuda.Event()
        size_block = 512
        block_ = (size_block, 1, 1)
        n_block = self.init_n_values_triu_extra // (size_block) + 1
        grid_all = (int(n_block / self.stride), 1)
        print "block = ", block_
        print "grid_all = ", grid_all
        #####################################################################################
        start.record()
        self.evaluate_likelihood(self.data,
                                 self.gpu_vect_frags.get_ptr(),
                                 self.gpu_collector_id_repeats,
                                 self.gpu_frag_dispatcher,
                                 self.gpu_sub_frag_id,
                                 self.gpu_rep_sub_frags_id,
                                 self.gpu_sub_frag_len_bp,
                                 self.gpu_sub_frag_accu,
                                 self.curr_likelihood,
                                 self.gpu_param_simu,
                                 self.init_n_values_triu,
                                 self.init_n_values_triu_extra,
                                 self.n_frags,
                                 self.init_n_sub_frags,
                                 self.mean_squared_frags_per_bin,
                                 block=block_, grid=grid_all, shared=0)

        #self.evaluate_likelihood(self.data,
        #                         self.gpu_vect_frags.get_ptr(),
        #                         self.gpu_collector_id_repeats,
        #                         self.gpu_frag_dispatcher,
        #                         self.gpu_sub_frag_id,
        #                         self.gpu_rep_sub_frags_id,
        #                         self.gpu_sub_frag_len_bp,
        #                         self.gpu_sub_frag_accu,
        #                         # self.gpu_full_expected_lin,
        #                         # self.gpu_full_expected_rep_lin,
        #                         self.curr_likelihood,
        #                         self.gpu_param_simu,
        #                         self.init_n_values_triu,
        #                         self.init_n_values_triu_extra,
        #                         self.n_frags,
        #                         self.init_n_sub_frags,
        #                         self.mean_squared_frags_per_bin,
        #                         block=block_, grid=grid_all, shared=0)
        end.record()
        end.synchronize()

        secs = start.time_till(end) * 1e-3
        print "CUDA clock execution timing (compute likelihood): ", secs
        #####################################################################################
        # size_block = 32
        # block_ = (size_block, 1, 1)
        # n_block = self.init_n_sub_values_triu // (size_block) + 1
        # grid_all = (int(n_block / self.stride), 1)
        # print "block = ", block_
        # print "grid_all = ", grid_all
        #
        # start.record()
        # self.fill_2d_mat(self.gpu_full_expected_lin,
        #                  self.gpu_2d_full_expected_mat,
        #                  self.init_n_sub_values_triu,
        #                  self.init_n_sub_frags,
        #                  block=block_, grid=grid_all, shared=0)
        # end.record()
        # end.synchronize()
        # self.cpu_2d_full_expected_mat = self.gpu_2d_full_expected_mat.get()
        # secs = start.time_till(end) * 1e-3
        # print "CUDA clock execution timing( fill 2d full matrix): ", secs
        # ####################################################################################
        # block_ = (size_block, 1, 1)
        # n_block = self.new_n_sub_values_triu // (size_block) + 1
        # grid_all = (int(n_block / self.stride), 1)
        # start.record()
        # self.fill_2d_mat(self.gpu_full_expected_rep_lin,
        #                  self.gpu_2d_full_expected_rep_mat,
        #                  self.new_n_sub_values_triu,
        #                  self.n_new_sub_frags,
        #                  block=block_, grid=grid_all, shared=0)
        # end.record()
        # end.synchronize()
        # self.cpu_2d_full_expected_rep_mat = self.gpu_2d_full_expected_rep_mat.get()
        # secs = start.time_till(end) * 1e-3
        # print "CUDA clock execution timing( fill 2d matrix): ", secs
        # ####################################################################################
        likelihood = ga.sum(self.curr_likelihood,dtype=np.float64).get()
        print "likelihood = ", likelihood
        return likelihood
        # ####################################################################################
        # plt.imshow(self.cpu_2d_full_expected_mat, interpolation='nearest', vmin=0, vmax=10,)
        # plt.show()


    def setup_texture(self,):
        # if self.perform_sub_sample:
        #     start = cuda.Event()
        #     end = cuda.Event()
        #
        #     size_block_x = 32
        #     size_block_y = 32
        #     n_blocks_x = int(self.sub_n_frags)//(size_block_x) + 1
        #     n_blocks_y = int(self.sub_n_frags)//(size_block_y) + 1
        #     self.grid = (n_blocks_x, n_blocks_y, 1)
        #     self.block = (size_block_x, size_block_y, 1)
        #
        #     new_data = np.zeros((self.sub_n_frags, self.sub_n_frags), dtype=np.float32)
        #     self.gpu_new_data = ga.to_gpu(new_data)
        #     start.record()
        #     self.gen_rand_mat(self.gpu_2d_hic_matrix, self.gpu_new_data, self.rng_states, np.int32(self.n_generators),
        #                       self.sub_n_frags, np.float32(self.sub_sample_factor),
        #                       block=self.block, grid=self.grid, shared=0)
        #     end.record()
        #     end.synchronize()
        #     secs = start.time_till(end) * 1e-3
        #     print "CUDA execution time (random matrix generation): ", secs
        #     self.data = self.gpu_new_data
        #     new_data = self.gpu_new_data.get()
        #     self.hic_matrix = new_data
        # else:
        #     self.data = self.gpu_2d_hic_matrix

        self.data = self.gpu_2d_hic_matrix

    def update_texture_4_sub(self, fact):
        if self.perform_sub_sample:
            start = cuda.Event()
            end = cuda.Event()

            size_block_x = 32
            size_block_y = 32
            n_blocks_x = int(self.sub_n_frags)//(size_block_x) + 1
            n_blocks_y = int(self.sub_n_frags)//(size_block_y) + 1
            self.grid = (n_blocks_x, n_blocks_y, 1)
            self.block = (size_block_x, size_block_y, 1)

            new_data = np.zeros((self.sub_n_frags, self.sub_n_frags), dtype=np.float32)
            self.gpu_new_data = ga.to_gpu(new_data)
            start.record()
            self.gen_rand_mat(self.gpu_2d_hic_matrix, self.gpu_new_data, self.rng_states, np.int32(self.n_generators),
                              self.sub_n_frags, np.float32(fact),
                              block=self.block, grid=self.grid, shared=0)
            end.record()
            end.synchronize()
            secs = start.time_till(end) * 1e-3
            print "CUDA execution time (random matrix generation): ", secs
            self.data = self.gpu_new_data
            new_data = self.gpu_new_data.get()
            self.hic_matrix = new_data
        else:
            self.data = self.gpu_2d_hic_matrix

        # self.data = self.gpu_2d_hic_matrix
        # tex_ref = self.module.get_texref("texData")
        # self.data.bind_to_texref_ext(tex_ref, channels=2, allow_double_hack=False, allow_offset=False)

    def meminfo(self, kernel):
        shared=kernel.shared_size_bytes
        regs=kernel.num_regs
        local=kernel.local_size_bytes
        const=kernel.const_size_bytes
        mbpt=kernel.max_threads_per_block
        print("""=MEM=\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d"""%(local,shared,regs,
                                                                                              const,mbpt))

    def loadProgram(self, filename):
        #read in the Cuda source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #create the program
        self.module = pycuda.compiler.SourceModule(fstr, no_extern_c=True, options=["-keep", "-m64", "-cubin"])
        # self.module = pycuda.compiler.SourceModule(fstr, no_extern_c=True,arch='sm_30',
        #                                            options=["-keep", "-m64", "-cubin"])

    def update_neighbourhood(self,):
        tmp_sorted = self.hic_matrix_sub_sampled.argsort(axis=1)
        sorted_neighbours = []
        for i in self.list_frag_to_sample:
            all_idx = tmp_sorted[i, :]
            pos = np.nonzero(all_idx == i)[0][0]
            line = list(all_idx)
            line.pop(pos)
            print 'filtering neighbourhood of :', i
            for j in self.list_to_pop_out:
                line = np.array(line)
                pos = np.nonzero(line == j)[0][0]
                line = list(line)
                line.pop(pos)

            sorted_neighbours.append(line)
        self.sorted_neighbours = np.array(sorted_neighbours)

    def pop_out_pop_in_4_mh(self, id_f_pop, id_f_ins, mode, max_id, forward):
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        # print 'max_id contig before pop out= ', max_id
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward

        self.pop_out(self.pop_gpu_vect_frags.get_ptr(), gpu_vect_frags.get_ptr(), self.pop_gpu_id_contigs,
                     np.int32(id_f_pop), max_id, self.n_frags,
                     block=block_, grid=grid_)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        max_id2 = np.int32(ga.max(self.pop_gpu_id_contigs).get())
        # print 'max_id contig after pop out= ', max_id2
        start.record()
        modif_vector = self.collector_gpu_vect_frags[mode]
        or_watson = np.int32(1)
        or_crick = np.int32(-1)
        # print "max_id from pop = ", max_id
        if mode == 0:
            self.simple_copy(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                             self.n_frags,
                             block=block_, grid=grid_)
        elif mode == 1:
            self.flip_frag(self.collector_gpu_vect_frags[mode].get_ptr(), gpu_vect_frags.get_ptr(),
                           np.int32(id_f_pop), self.n_frags,
                           block=block_, grid=grid_)
        elif mode == 2:
            self.pop_in_3(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_frags,
                         block=block_, grid=grid_)
        elif mode == 3:
            self.pop_in_3(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_frags,
                          block=block_, grid=grid_)

        elif mode == 4:
            self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_frags,
                         block=block_, grid=grid_)
        elif mode == 5:
            self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_frags,
                         block=block_, grid=grid_)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3

    def split_4_mh(self, id_fA, max_id, forward):
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        start = cuda.Event()
        end = cuda.Event()
        mode = 0
        id_start_split = 6
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward

        for upstreamfA in range(0, 2):
            start.record()
            vect_frags_2_modif = self.collector_gpu_vect_frags[id_start_split + mode]
            self.split(vect_frags_2_modif.get_ptr(), gpu_vect_frags.get_ptr(), self.trans1_gpu_id_contigs,
                       np.int32(id_fA), np.int32(upstreamfA), max_id, self.n_frags,
                       block=block_, grid=grid_)
            mode += 1

    def paste_4_mh(self, id_fA, id_fB, max_id, forward):
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward

        vect_frags_2_modif = self.collector_gpu_vect_frags[8]
        # vect_frags_2_modif = self.collector_gpu_vect_frags[8]

        start.record()
        is_extremity_fA = (gpu_vect_frags.prev[id_fA] == -1) or (gpu_vect_frags.next[id_fA] == -1)
        is_extremity_fB = (gpu_vect_frags.prev[id_fB] == -1) or (gpu_vect_frags.next[id_fB] == -1)
        if  is_extremity_fA and is_extremity_fB:
            self.paste(vect_frags_2_modif.get_ptr(), gpu_vect_frags.get_ptr(),
                       np.int32(id_fA), np.int32(id_fB), max_id, self.n_frags,
                       block=block_, grid=grid_)
        else:
            self.simple_copy(vect_frags_2_modif.get_ptr(), gpu_vect_frags.get_ptr(),
                             self.n_frags,
                             block=block_, grid=grid_)
        end.record()
        end.synchronize()

    def pop_out_pop_in(self, id_f_pop, id_f_ins, mode, max_id):
        size_block = 128
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        # print 'max_id contig before pop out= ', max_id
        start = cuda.Event()
        end = cuda.Event()
        start.record()

        self.pop_out(self.pop_gpu_vect_frags.get_ptr(), self.gpu_vect_frags.get_ptr(), self.pop_gpu_id_contigs,
                     np.int32(id_f_pop), max_id, self.n_new_frags,
                     block=block_, grid=grid_)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        max_id2 = np.int32(ga.max(self.pop_gpu_id_contigs).get())
        # print 'max_id contig after pop out= ', max_id2
        start.record()
        modif_vector = self.collector_gpu_vect_frags[mode]
        or_watson = np.int32(1)
        or_crick = np.int32(-1)
        # print "max_id from pop = ", max_id
        if mode == 0: # eject frag
            self.simple_copy(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                             self.n_new_frags,
                             block=block_, grid=grid_)
        elif mode == 1:
            self.flip_frag(self.collector_gpu_vect_frags[mode].get_ptr(), self.gpu_vect_frags.get_ptr(),
                           np.int32(id_f_pop), self.n_new_frags,
                           block=block_, grid=grid_)
        elif mode == 2:
            self.pop_in_1(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_new_frags,
                         block=block_, grid=grid_)
        elif mode == 3:
            self.pop_in_1(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_new_frags,
                         block=block_, grid=grid_)
        elif mode == 4:
            self.pop_in_2(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_new_frags,
                         block=block_, grid=grid_)
        elif mode == 5:
            self.pop_in_2(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_new_frags,
                         block=block_, grid=grid_)
        elif mode == 6:
            self.pop_in_3(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_new_frags,
                         block=block_, grid=grid_)
        elif mode == 7:
            self.pop_in_3(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_new_frags,
                          block=block_, grid=grid_)

        # elif mode == 8:
        #     self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
        #                   np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_watson, self.n_frags,
        #                  block=block_, grid=grid_)
        # elif mode == 9:
        #     self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
        #                   np.int32(id_f_pop), np.int32(id_f_ins), max_id2, or_crick, self.n_frags,
        #                  block=block_, grid=grid_)

        elif mode == 8:
            self.swap_activity(self.collector_gpu_vect_frags[mode].get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                          np.int32(id_f_pop), max_id2, self.n_new_frags,
                          block=block_, grid=grid_)


        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3

    def transloc(self, id_fA, id_fB, max_id):
        size_block = 128
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        start = cuda.Event()
        end = cuda.Event()
        mode = 0
        # id_start_transloc = 8
        id_start_transloc = 9
        for upstreamfA in range(0, 2):
            start.record()
            self.split(self.trans1_gpu_vect_frags.get_ptr(), self.gpu_vect_frags.get_ptr(), self.trans1_gpu_id_contigs,
                       np.int32(id_fA), np.int32(upstreamfA), max_id, self.n_new_frags,
                       block=block_, grid=grid_)
            end.record()
            end.synchronize()
            for upstreamfB in range(0, 2):
                max_id1 = np.int32(ga.max(self.trans1_gpu_id_contigs).get())
                # print 'max_id1 = ', max_id1
                start.record()
                self.split(self.trans2_gpu_vect_frags.get_ptr(), self.trans1_gpu_vect_frags.get_ptr(),
                           self.trans2_gpu_id_contigs,
                           np.int32(id_fB), np.int32(upstreamfB), max_id1, self.n_new_frags,
                           block=block_, grid=grid_)
                end.record()
                end.synchronize()
                max_id2 = np.int32(ga.max(self.trans2_gpu_id_contigs).get())
                # print 'max_id2 = ', max_id2
                curr_vect_trans = self.collector_gpu_vect_frags[id_start_transloc + mode]
                start.record()
                # self.simple_copy(curr_vect_trans.get_ptr(), self.trans2_gpu_vect_frags.get_ptr(), self.n_frags,
                #                  block=block_, grid=grid_)
                self.paste(curr_vect_trans.get_ptr(), self.trans2_gpu_vect_frags.get_ptr(),
                           np.int32(id_fA), np.int32(id_fB), max_id2, self.n_new_frags,
                           block=block_, grid=grid_)
                end.record()
                end.synchronize()
                mode += 1


    def transloc_4_mh(self, id_fA, id_fB, max_id, forward):
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        start = cuda.Event()
        end = cuda.Event()
        mode = 0
        id_start_transloc = 9
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward

        for upstreamfA in range(0, 2):

            start.record()
            self.split(self.trans1_gpu_vect_frags.get_ptr(), gpu_vect_frags.get_ptr(), self.trans1_gpu_id_contigs,
                       np.int32(id_fA), np.int32(upstreamfA), max_id, self.n_frags,
                       block=block_, grid=grid_)
            end.record()
            end.synchronize()
            for upstreamfB in range(0, 2):
                # print "mode = ", mode
                max_id1 = np.int32(ga.max(self.trans1_gpu_id_contigs).get())
                if upstreamfB == 0 :
                    if gpu_vect_frags.next[id_fB] == -1:
                        perform_transloc = True
                    else:
                        perform_transloc = False
                else:
                    if gpu_vect_frags.prev[id_fB] == -1:
                        perform_transloc = True
                    else:
                        perform_transloc = False
                start.record()
                curr_vect_trans = self.collector_gpu_vect_frags[id_start_transloc + mode]
                if perform_transloc:
                    self.split(self.trans2_gpu_vect_frags.get_ptr(), self.trans1_gpu_vect_frags.get_ptr(),
                               self.trans2_gpu_id_contigs,
                               np.int32(id_fB), np.int32(upstreamfB), max_id1, self.n_frags,
                               block=block_, grid=grid_)
                    end.record()
                    end.synchronize()
                    max_id2 = np.int32(ga.max(self.trans2_gpu_id_contigs).get())
                    # print 'max_id2 = ', max_id2
                    start.record()
                    self.paste(curr_vect_trans.get_ptr(), self.trans2_gpu_vect_frags.get_ptr(),
                               np.int32(id_fA), np.int32(id_fB), max_id2, self.n_frags,
                               block=block_, grid=grid_)
                else:
                    self.simple_copy(curr_vect_trans.get_ptr(), gpu_vect_frags.get_ptr(), self.n_frags,
                                     block=block_, grid=grid_)
                end.record()
                end.synchronize()
                mode += 1



    def diagnosis(self, c, id_fA, id_fB, id_mut):
        c.copy_from_gpu()
        list_idc = np.unique(c.id_c)
        list_start = np.nonzero(c.start_bp == 0)[0]
        for ele in list_start:
            len_contig = c.l_cont[ele]
            cur_f = ele
            for f in range(1, len_contig):
                cur_f = c.next[cur_f]
            extrem = np.int32(cur_f)
            for f in range(1, len_contig):
                cur_f = c.prev[cur_f]
            if c.circ[ele] == 1:
                if extrem != c.prev[ele]:
                    print "problem!!!!!! @ contig :", c.id_c[ele]
                    print "id frag prob = ", ele
                    print "id_fA = ", id_fA
                    print "id_fB = ", id_fB
                    print "id_mut = ", id_mut
                    raw_input("what shoud I do?")

            if cur_f != ele:
                print "problem!!!!!! @ contig :", c.id_c[ele]
                print "id_fA = ", id_fA
                print "id_fB = ", id_fB
                print "id_mut = ", id_mut
                raw_input("what shoud I do?")


    def new_perform_modificationS(self, id_fA, id_fB, max_id, is_first):
        for mode in xrange(0, 9):
            self.pop_out_pop_in(id_fA, id_fB, mode, max_id)
        self.transloc(id_fA, id_fB, max_id)
        # for mode in xrange(14, self.n_tmp_struct):
        #     self.local_flip(id_fA, mode, max_id)
        # self.all_pop_out_pop_in(id_fA, id_fB, max_id, is_first)
        # tic_fillB = time.time()
        # self.all_transloc(id_fA, id_fB, max_id, is_first)
        # print "all_pop out time execution  = ", time.time() - tic_fillB

    def local_flip(self, id_fA, mode, max_id):
        # mode = 12
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        local_delta = mode - 11
        vect_frags = self.gpu_vect_frags
        vect_frags.copy_from_gpu()

        pos_fA = vect_frags.pos[id_fA]
        id_contig_A = vect_frags.id_c[id_fA]
        len_contig_A = vect_frags.l_cont[id_fA]

        id_f_in_contig_A = np.nonzero(vect_frags.id_c == id_contig_A)[0]
        neighbours = id_f_in_contig_A
        pos_neighbours = vect_frags.pos[neighbours]

        arg_sort_id = np.argsort(pos_neighbours)

        ordered_neighbours = neighbours[arg_sort_id]
        orientations_neighbours = vect_frags.ori[ordered_neighbours]
        id_up = max(pos_fA - local_delta, 0)
        id_down = min(pos_fA + local_delta, len_contig_A - 1)
        # print "id_fA = ", id_fA
        # print "pos_fA = ", pos_fA
        # print "pos up = ", id_up
        # print "pos down = ", id_down

        start.record()
        self.simple_copy(self.scrambled_gpu_vect_frags.get_ptr(), self.gpu_vect_frags.get_ptr(),
                         self.n_new_frags,
                         block=block_, grid=grid_)
        end.record()
        end.synchronize()
        # print "ordered neighbours = ", ordered_neighbours
        for i in range(id_up, id_down + 1):
            id_fB = ordered_neighbours[i]
            if id_fB != id_fA:
                start.record()
                self.pop_out(self.pop_gpu_vect_frags.get_ptr(), self.scrambled_gpu_vect_frags.get_ptr(), self.pop_gpu_id_contigs,
                             np.int32(id_fB), max_id, self.n_new_frags,
                             block=block_, grid=grid_)
                end.record()
                end.synchronize()
                start.record()
                self.simple_copy(self.scrambled_gpu_vect_frags.get_ptr(), self.pop_gpu_vect_frags.get_ptr(),
                                 self.n_new_frags,
                                 block=block_, grid=grid_)
                end.record()
                end.synchronize()
                self.scrambled_gpu_vect_frags.copy_from_gpu()
                max_id = self.scrambled_gpu_vect_frags.id_c.max()

        for j in range(id_down, pos_fA, -1):
            id_fB = ordered_neighbours[j]
            ori_fB = orientations_neighbours[j] * -1
            # print "id_fB = ", id_fB
            start.record()
            self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(), self.scrambled_gpu_vect_frags.get_ptr(),
                          np.int32(id_fB), np.int32(id_fA), max_id, np.int32(ori_fB), self.n_new_frags,
                         block=block_, grid=grid_)
            end.record()
            end.synchronize()
            start.record()
            self.simple_copy(self.scrambled_gpu_vect_frags.get_ptr(), self.collector_gpu_vect_frags[mode].get_ptr(),
                             self.n_new_frags,
                             block=block_, grid=grid_)
            end.record()
            end.synchronize()
            self.scrambled_gpu_vect_frags.copy_from_gpu()
            max_id = self.scrambled_gpu_vect_frags.id_c.max()
        # print "insert left ok"
        for j in range(id_up, pos_fA):
            id_fB = ordered_neighbours[j]
            ori_fB = orientations_neighbours[j] * -1
            # print "id_fB = ", id_fB
            start.record()
            self.pop_in_3(self.collector_gpu_vect_frags[mode].get_ptr(), self.scrambled_gpu_vect_frags.get_ptr(),
                          np.int32(id_fB), np.int32(id_fA), max_id, np.int32(ori_fB), self.n_new_frags,
                         block=block_, grid=grid_)
            end.record()
            end.synchronize()
            start.record()
            self.simple_copy(self.scrambled_gpu_vect_frags.get_ptr(), self.collector_gpu_vect_frags[mode].get_ptr(),
                             self.n_new_frags,
                             block=block_, grid=grid_)
            end.record()
            end.synchronize()
            self.scrambled_gpu_vect_frags.copy_from_gpu()
            max_id = self.scrambled_gpu_vect_frags.id_c.max()

        start.record()
        self.flip_frag(self.collector_gpu_vect_frags[mode].get_ptr(), self.scrambled_gpu_vect_frags.get_ptr(),
               np.int32(id_fA), self.n_new_frags,
               block=block_, grid=grid_)
        end.record()
        end.synchronize()

    def test_copy_struct(self, id_fA, id_f_sampled, mode, max_id):
        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        if mode < 9:
            self.pop_out_pop_in(id_fA, id_f_sampled, mode, max_id)
        elif mode < 13:
            self.transloc(id_fA, id_f_sampled, max_id)
        elif mode >= 12:
            self.local_flip(id_fA, mode, max_id)
        # else:
        #     self.local_scramble(id_fA, max_id)
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        # print 'mode = ', mode
        sampled_vect_frags = self.collector_gpu_vect_frags[mode]
        # sampled_vect_frags.copy_from_gpu()
        # plt.plot(sampled_vect_frags.id_c)
        # plt.show()

        self.copy_vect(self.gpu_vect_frags.get_ptr(), sampled_vect_frags.get_ptr(), self.gpu_id_contigs, self.n_new_frags,
                       block=block_, grid=grid_, shared=0)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3


    def setup_rippe_parameters_4_simu(self, kuhn, lm, slope, d, val_inter, d_max):

        kuhn = np.float32(kuhn)
        lm = np.float32(lm)
        c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
        slope = np.float32(slope)
        d = np.float32(d)
        d_max = np.float32(d_max)
        val_inter = np.float32(val_inter)
        parameters = [kuhn, lm, slope, d]
        Rippe = lambda param, dist: (0.53 * (param[0] ** -3.) * np.power((param[1] * dist/param[0]), (param[2])) *
                                     np.exp((param[3] - 2) / ((np.power((param[1] * dist / param[0]), 2) + param[3]))))
        rippe_inter = Rippe(parameters, d_max)
        fact = val_inter / rippe_inter
        p = np.array([(kuhn, lm, c1, slope, d, d_max, fact, val_inter)], dtype=self.param_simu_T)
        return p

    def setup_rippe_parameters(self, param, d_max):

        kuhn, lm, slope, d, fact = param
        kuhn = np.float32(kuhn)
        lm = np.float32(lm)
        c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
        slope = np.float32(slope)
        d = np.float32(d)
        fact = np.float32(fact)
        d_max = np.float32(d_max)
        p = np.array([(kuhn, lm, c1, slope, d, d_max, fact, self.mean_value_trans)], dtype=self.param_simu_rippe)
        return p

    def setup_model_parameters(self, param, d_max):

        d0, d1, alpha_0, alpha_1, alpha_2, fact = param
        d0 = np.float32(d0)
        d1 = np.float32(d1)
        d_max = np.float32(d_max)
        alpha_0 = np.float32(alpha_0)
        alpha_1 = np.float32(alpha_1)
        alpha_2 = np.float32(alpha_2)
        fact = np.float32(fact)
        p = np.array([(d0, d1, d_max, alpha_0, alpha_1, alpha_2, fact, self.mean_value_trans)], dtype=self.param_simu_exp)
        return p

    def estimate_parameters(self, max_dist_kb, size_bin_kb):
        """
        estimation by least square optimization of Rippe parameters on the experimental data
        :param max_dist_kb:
        :param size_bin_kb:
        """
        print "estimation of the parameters of the model"
        self.bins = np.arange(size_bin_kb, max_dist_kb + size_bin_kb, size_bin_kb)
        # print "bins = ", self.bins
        self.mean_contacts = np.zeros_like(self.bins, dtype=np.float32)
        self.dict_collect = dict()
        self.gpu_vect_frags.copy_from_gpu()
        epsi = 10**-10
        for k in self.bins:
            self.dict_collect[k] = []
        for i in xrange(0, self.sub_n_frags - 1):
            id_c_i = self.S_o_A_sub_frags['id_c'][i]
            start_fi = self.S_o_A_sub_frags['start_bp'][i]
            len_fi = self.S_o_A_sub_frags['len_bp'][i]
            pos_i = self.S_o_A_sub_frags['pos'][i]
            for j in xrange(i + 1, self.sub_n_frags):
                id_c_j = self.S_o_A_sub_frags['id_c'][j]
                if id_c_i == id_c_j:
                    ncontacts = self.hic_matrix[i, j]
                    start_fj = self.S_o_A_sub_frags['start_bp'][j]
                    len_fj = self.S_o_A_sub_frags['len_bp'][j]
                    pos_j = self.S_o_A_sub_frags['pos'][j]
                    if pos_i < pos_j:
                        d = ((start_fj - start_fi - len_fi) + (len_fi + len_fj)/2.)/1000.
                    else:
                        d = ((start_fi - start_fj - len_fj) + (len_fj + len_fi)/2.)/1000.
                    if d < max_dist_kb:
                        id_bin = d / size_bin_kb
                        self.dict_collect[self.bins[id_bin]].append(ncontacts)
        for id_bin in xrange(0, len(self.bins)):
            k = self.bins[id_bin]
            tmp = np.mean(self.dict_collect[k])
            if np.isnan(tmp) or tmp == 0:
            # if np.isnan(tmp):
                self.mean_contacts[id_bin] = epsi
            else:
                self.mean_contacts[id_bin] = tmp

        p, self.y_estim = opti.estimate_param_rippe(self.mean_contacts, self.bins)
        ##########################################
        print "p from estimate parameters  = ", p
        # p = list(p[0])
        # p[3] = 2
        # p = tuple(p)
        ##########################################
        fit_param = p
        print "mean value trans = ", self.mean_value_trans
        estim_max_dist = opti.estimate_max_dist_intra(fit_param, self.mean_value_trans)
        self.param_simu = self.setup_rippe_parameters(fit_param, estim_max_dist)
        # fig = plt.figure()
        # plt.loglog(self.bins, self.mean_contacts, '-*b')
        # plt.loglog(self.bins, self.y_estim, '-*r')
        # plt.xlabel("genomic distance (kb)")
        # plt.ylabel("frequency of contact")
        # plt.title(r'$\mathrm{Frequency\ of\ contact\ versus\ genomic\ distance\ (data):}\ slope=%.3f,\ max\ cis\ distance(kb)=%.3f\ d=%.3f\ scale\ factor=%.3f\ $' %( self.param_simu['slope'], estim_max_dist, self.param_simu['d'], self.param_simu['fact']))
        # plt.legend(['obs', 'fit'])
        # plt.show()
        self.gpu_param_simu = cuda.mem_alloc(self.param_simu.nbytes)
        self.gpu_param_simu_test = cuda.mem_alloc(self.param_simu.nbytes)

        cuda.memcpy_htod(self.gpu_param_simu, self.param_simu)

    def estimate_parameters_rv(self, max_dist_kb, size_bin_kb):
        """
        estimation by least square optimization of Rippe parameters on the experimental data
        :param max_dist_kb:
        :param size_bin_kb:
        """
        self.bins = np.arange(size_bin_kb, max_dist_kb + size_bin_kb, size_bin_kb, dtype=np.float32)
        # print "bins = ", self.bins
        self.mean_contacts = np.zeros_like(self.bins, dtype=np.float32)
        self.dict_collect = dict()
        self.gpu_vect_frags.copy_from_gpu()
        for k in self.bins:
            self.dict_collect[k] = []
        for i in xrange(0, self.sub_n_frags - 1):
            id_c_i = self.S_o_A_sub_frags['id_c'][i]
            start_fi = self.S_o_A_sub_frags['start_bp'][i]
            len_fi = self.S_o_A_sub_frags['len_bp'][i]
            pos_i = self.S_o_A_sub_frags['pos'][i]
            for j in xrange(i + 1, self.sub_n_frags):
                id_c_j = self.S_o_A_sub_frags['id_c'][j]
                if id_c_i == id_c_j:
                    ncontacts = self.hic_matrix[i, j]
                    start_fj = self.S_o_A_sub_frags['start_bp'][j]
                    len_fj = self.S_o_A_sub_frags['len_bp'][j]
                    pos_j = self.S_o_A_sub_frags['pos'][j]
                    if pos_i < pos_j:
                        d = ((start_fj - start_fi - len_fi) + (len_fi + len_fj)/2.)/1000.
                    else:
                        d = ((start_fi - start_fj - len_fj) + (len_fj + len_fi)/2.)/1000.
                    if d < max_dist_kb:
                        id_bin = d / size_bin_kb
                        self.dict_collect[self.bins[id_bin]].append(ncontacts)
        for id_bin in xrange(0, len(self.bins)):
            k = self.bins[id_bin]
            self.mean_contacts[id_bin] = np.mean(self.dict_collect[k])
        p, self.y_estim = opti_rv.estimate_param_hic(self.mean_contacts, self.bins)
        print "val inter = ", self.mean_value_trans
        fit_param = p[0]

        estim_max_dist = opti_rv.estimate_max_dist_intra(fit_param, self.mean_value_trans)
        plt.axvspan(fit_param[0], fit_param[0])
        plt.axvspan(fit_param[1], fit_param[1])
        plt.axvspan(estim_max_dist, estim_max_dist)

        plt.loglog(self.bins, self.mean_contacts, '-*b')
        print "bin min = ", self.bins.min()
        print "bin max = ", self.bins.max()
        plt.loglog(np.arange(self.bins.min(), self.bins.max(), 5), self.y_estim, '-*r')
        plt.ylim(self.mean_value_trans*0.01, fit_param[5])
        plt.legend(['obs', 'fit'])
        plt.show()

        self.param_simu = self.setup_model_parameters(fit_param, estim_max_dist)
        print "param simu = ", self.param_simu
        raw_input("alors ?")
        self.gpu_param_simu = cuda.mem_alloc(self.param_simu.nbytes)
        cuda.memcpy_htod(self.gpu_param_simu, self.param_simu)


    def simulate_rippe_contacts(self,):
        # max_dist_kb = 1000
        # size_bin_kb = 10
        # self.estimate_parameters(max_dist_kb, size_bin_kb)


        # self.param_simu = self.setup_rippe_parameters_4_simu(kuhn, lm, slope, d, val_inter, d_max)
        # self.gpu_param_simu = cuda.mem_alloc(self.param_simu.nbytes)
        # cuda.memcpy_htod(self.gpu_param_simu, self.param_simu)
        size_block = 128
        block_ = (size_block, 1, 1)
        n_blocks = int(self.init_n_values_triu_extra // size_block + 1)
        grid_ = (int(n_blocks // self.stride + 1), 1)
        # grid_ = (int(self.n_values_triu // size_block + 1), 1)
        print "block = ", block_
        print "grid = ", grid_
        start = cuda.Event()
        end = cuda.Event()

        start.record()
        print "hic matrix shape = ", self.hic_matrix.shape
        print "sub sampled hic matrix shape = ", self.hic_matrix_sub_sampled.shape


        out_simu = np.ones((self.hic_matrix.shape), dtype=np.float32)
        out_sub_simu = np.zeros((self.hic_matrix_sub_sampled.shape), dtype=np.float32)
        self.gpu_2d_simu_hic_matrix = ga.to_gpu(ary=out_simu)
        self.gpu_2d_simu_sub_sample_matrix = ga.to_gpu(ary=out_sub_simu)

        self.simulate_data_2d(self.gpu_vect_frags.get_ptr(),
                              self.gpu_collector_id_repeats,
                              self.gpu_frag_dispatcher,
                              self.gpu_sub_frag_id,
                              self.gpu_rep_sub_frags_id,
                              self.gpu_sub_frag_len_bp,
                              self.gpu_sub_frag_accu,
                              self.gpu_2d_simu_sub_sample_matrix,
                              self.gpu_2d_simu_hic_matrix,
                              self.gpu_param_simu,
                              self.init_n_values_triu,
                              self.init_n_values_triu_extra,
                              self.n_frags,
                              self.init_n_sub_frags,
                              self.mean_squared_frags_per_bin,
                              self.rng_states, np.int32(self.n_generators,),
                              block=block_, grid=grid_, shared=0)

        end.record()
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        print "CUDA clock execution timing( simulate data): ", secs
        start.record()

        self.gpu_2d_simu_hic_matrix.get(ary=out_simu)

        self.gpu_2d_simu_sub_sample_matrix.get(ary=out_sub_simu)
        self.hic_matrix_sub_sampled = out_sub_simu.T + out_sub_simu
        self.cpu_simu_matrix = out_simu.T + out_simu
        self.hic_matrix = self.cpu_simu_matrix
        self.gpu_2d_hic_matrix = ga.to_gpu(ary=self.cpu_simu_matrix)
        end.record()
        end.synchronize()

        self.data = self.gpu_2d_hic_matrix
        self.define_neighbourhood()

    def display_modif_vect(self, id_fA, id_fB, mode, thresh):

        self.gpu_vect_frags.copy_from_gpu()
        max_id = self.gpu_vect_frags.id_c.max()
        self.new_perform_modificationS(id_fA, id_fB, max_id, True)
        size_block = 128
        block_ = (size_block, 1, 1)
        n_blocks = int(self.init_n_values_triu_extra // size_block + 1)
        grid_ = (int(n_blocks // self.stride + 1), 1)
        # grid_ = (int(self.n_values_triu // size_block + 1), 1)
        print "block = ", block_
        print "grid = ", grid_
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        init_data = np.ones((self.hic_matrix.shape), dtype=np.float32)
        self.gpu_2d_simu_hic_matrix = ga.to_gpu(ary=init_data)
        self.gpu_2d_simu_sub_sample_matrix = ga.to_gpu(ary=np.zeros(self.hic_matrix_sub_sampled.shape, dtype=np.float32))
        if mode >= 0:
            v = self.collector_gpu_vect_frags[mode].get_ptr()
        else:
            v = self.gpu_vect_frags.get_ptr()
        self.simulate_data_2d(v,
                              self.gpu_collector_id_repeats,
                              self.gpu_frag_dispatcher,
                              self.gpu_sub_frag_id,
                              self.gpu_rep_sub_frags_id,
                              self.gpu_sub_frag_len_bp,
                              self.gpu_sub_frag_accu,
                              self.gpu_2d_simu_sub_sample_matrix,
                              self.gpu_2d_simu_hic_matrix,
                              self.gpu_param_simu,
                              self.init_n_values_triu,
                              self.init_n_values_triu_extra,
                              self.n_frags,
                              self.init_n_sub_frags,
                              self.mean_squared_frags_per_bin,
                              self.rng_states, np.int32(self.n_generators,),
                              block=block_, grid=grid_, shared=0)

        end.record()
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        print "CUDA clock execution timing( simulate data): ", secs
        start.record()
        out = np.zeros((self.hic_matrix.shape) ,dtype=np.float32)
        self.gpu_2d_simu_hic_matrix.get(ary=out)
        self.cpu_simu_matrix = out + out.T
        plt.imshow(self.cpu_simu_matrix, vmin=0, vmax=thresh, interpolation='nearest')
        plt.colorbar()
        plt.show()

    def compute_likelihood(self, gpu_vect_frags, likelihood_vect):
        start = cuda.Event()
        end = cuda.Event()



        size_block = 512
        block_ = (size_block, 1, 1)
        # stride = 10
        n_block = self.init_n_values_triu_extra // (size_block) + 1
        grid_all = (int(n_block / self.stride), 1)
        start.record()
        # print "dim block = ", block_
        # print "dim grid =", grid_all
        self.evaluate_likelihood(self.data,
                                 gpu_vect_frags.get_ptr(),
                                 self.gpu_collector_id_repeats,
                                 self.gpu_frag_dispatcher,
                                 self.gpu_sub_frag_id,
                                 self.gpu_rep_sub_frags_id,
                                 self.gpu_sub_frag_len_bp,
                                 self.gpu_sub_frag_accu,
                                 likelihood_vect,
                                 self.gpu_param_simu,
                                 self.init_n_values_triu,
                                 self.init_n_values_triu_extra,
                                 self.n_frags,
                                 self.init_n_sub_frags,
                                 self.mean_squared_frags_per_bin,
                                 block=block_, grid=grid_all, shared=0)
        end.record()
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        # print "CUDA clock execution timing( likelihood full): ", secs
        likelihood_star = ga.sum(likelihood_vect, dtype=np.float64).get()

        return likelihood_star


    def insert_repeats(self, id_f_ins):
        for id in range(0, self.n_new_frags):
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            if self.gpu_vect_frags.rep[id] == 1:
                print "id repeats = ", id
                mode = 7
                self.test_copy_struct(id, id_f_ins, mode, max_id)

    def modify_genome(self, n):
        list_breaks = np.random.choice(self.n_new_frags, n * 2, replace=False)
        list_modes = np.random.choice(self.n_tmp_struct, n, replace=True)
        for i in range(0, n):
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            self.test_copy_struct(list_breaks[2*i],list_breaks[2*i + 1], list_modes[i], max_id)
            self.gpu_vect_frags.copy_from_gpu()
            c = self.gpu_vect_frags
            if np.any(c.pos<0) or np.any(c.l_cont < 0) or np.any(c.l_cont_bp < 0) \
               or np.any(c.start_bp < 0) or np.any( c.l_cont_bp - c.start_bp <= 0) or np.any((c.start_bp !=0)*(c.pos ==0)) \
                or np.any((c.start_bp ==0)*(c.pos !=0)) or np.any(c.next == c.id) or np.any(c.prev == c.id):
                print 'problem!!!!'
                raw_input('what shoud I do????')
            if np.any(c.l_cont == 0) or np.any(c.l_cont_bp == 0):
                print 'problem null contig !!!!'
                raw_input('what shoud I do????')

    def explode_genome(self,dt):
        for i in range(0, self.n_new_frags):
            self.modify_gl_cuda_buffer(i, dt)
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            self.test_copy_struct(i, 0, 0, max_id)
            self.gpu_vect_frags.copy_from_gpu()
            c = self.gpu_vect_frags
            self.gl_window.remote_update()
            if np.any(c.pos<0) or np.any(c.l_cont < 0) or np.any(c.l_cont_bp < 0) \
               or np.any(c.start_bp < 0) or np.any( c.l_cont_bp - c.start_bp <= 0) or np.any((c.start_bp !=0)*(c.pos ==0)) \
                or np.any((c.start_bp ==0)*(c.pos !=0)) or np.any(c.next == c.id) or np.any(c.prev == c.id):
                print 'problem!!!!'
                raw_input('what shoud I do????')
            if np.any(c.l_cont == 0) or np.any(c.l_cont_bp == 0):
                print 'problem null contig !!!!'
                raw_input('what shoud I do????')
        print "genome exploded"
        print "max id = ", max_id

    def apply_replay_simu(self, id_fA, id_fB, op_sampled, dt):

        # n_modif = len(op_sampled)
        # for i in xrange(0, n_modif):
        #     self.modify_gl_cuda_buffer(i, dt)
        #     self.gpu_vect_frags.copy_from_gpu()
        #     max_id = self.gpu_vect_frags.id_c.max()
        #     self.test_copy_struct(id_fA[i], id_fB[i], op_sampled[i], max_id)
        #     self.gpu_vect_frags.copy_from_gpu()
        #     c = self.gpu_vect_frags
        #     self.gl_window.remote_update()


        self.modify_gl_cuda_buffer(id_fA, dt)
        self.gpu_vect_frags.copy_from_gpu()
        max_id = self.gpu_vect_frags.id_c.max()
        self.test_copy_struct(id_fA, id_fB, op_sampled, max_id)
        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        self.gl_window.remote_update()


    def display_current_matrix(self, file):
        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        self.gpu_id_contigs.get(ary=self.cpu_id_contigs)
        pos_frag = np.copy(self.gpu_vect_frags.pos)
        list_id_frags = np.copy(self.gpu_vect_frags.id_d)
        list_id = np.copy(self.cpu_id_contigs)
        list_activ = np.copy(self.gpu_vect_frags.activ)
        unique_contig_id = np.unique(list_id)
        dict_contig = dict()
        full_order = []
        full_order_high = []
        for k in unique_contig_id:
            dict_contig[k] = []
            id_pos = np.ix_(list_id == k)
            is_activ = list_activ[id_pos]
            if np.all(is_activ == 1):
                tmp_ord = np.argsort(pos_frag[id_pos])
                ordered_frag = list_id_frags[id_pos[0][tmp_ord]]
                dict_contig[k].extend(ordered_frag)
                full_order.extend(ordered_frag)
                for i in ordered_frag:
                    ori = c.ori[i]
                    v_high_id_all = list(self.np_sub_frags_id[i])
                    # print "v_high_id_all = ", v_high_id_all
                    # print "sub id =", range(0,v_high_id_all[3])
                    v_high_id = v_high_id_all[:v_high_id_all[3]]
                    id_2_push = list(v_high_id)
                    if ori == -1:
                        id_2_push.reverse()
                    full_order_high.extend(id_2_push)
        # fig = plt.figure(figsize=(10,10))
        val_max = self.hic_matrix_sub_sampled.max() * 0.01
        # plt.imshow(self.hic_matrix_sub_sampled[np.ix_(full_order,full_order)], vmin=0, vmax=50, interpolation='nearest')
        # fig.savefig(file)
        # plt.show()
        # plt.close()
        # plt.figure()
        # plt.imshow(self.hic_matrix[np.ix_(full_order_high, full_order_high)], vmin=0, vmax=20,
        #            interpolation='nearest')
        # plt.show()
        output_image = Image.fromarray(self.hic_matrix[np.ix_(full_order_high, full_order_high)])
        output_image.save(file)
        return full_order, dict_contig, full_order_high

    def genome_content(self):

        self.gpu_vect_frags.copy_from_gpu()
        self.gpu_id_contigs.get(ary=self.cpu_id_contigs)
        pos_frag = np.copy(self.gpu_vect_frags.pos)
        next_frag = np.copy(self.gpu_vect_frags.next)
        prev_frag = np.copy(self.gpu_vect_frags.prev)
        start_bp = np.copy(self.gpu_vect_frags.start_bp)
        list_id_frags = np.copy(self.gpu_vect_frags.id_d)
        list_activ = np.copy(self.gpu_vect_frags.activ)
        list_id = np.copy(self.gpu_vect_frags.id_c)
        unique_contig_id = np.unique(list_id)
        dict_contig = dict()

        full_order = []
        for k in unique_contig_id:
            dict_contig[k] = dict()
            dict_contig[k]['id'] = []
            dict_contig[k]['pos'] = []
            dict_contig[k]['next'] = []
            dict_contig[k]['prev'] = []
            dict_contig[k]['start_bp'] = []
            dict_contig[k]['id_c'] = []


            id_pos = np.ix_(list_id == k)[0]
            if np.all(list_activ[id_pos] == 1):
                # print "len id_pos = ", id_pos[0]
                tmp_ord = np.argsort(pos_frag[id_pos])
                # print "tmp_ord = ", tmp_ord
                l_start = start_bp[id_pos[tmp_ord]]
                l_pos = pos_frag[id_pos[tmp_ord]]
                l_id_c = list_id[id_pos[tmp_ord]]
                l_next = next_frag[id_pos[tmp_ord]]
                l_prev = prev_frag[id_pos[tmp_ord]]
                ordered_frag = list_id_frags[id_pos[tmp_ord]]
                dict_contig[k]['id'].extend(ordered_frag)
                dict_contig[k]['pos'].extend(l_pos)
                dict_contig[k]['start_bp'].extend(l_start)
                dict_contig[k]['id_c'].extend(l_id_c)
                dict_contig[k]['prev'].extend(l_prev)
                dict_contig[k]['next'].extend(l_next)
                full_order.extend(ordered_frag)
        return full_order, dict_contig


    def load_gl_cuda_vbo(self,):
         #CUDA Ressorces
        self.pos_vbo.bind()
        self.gpu_pos = cudagl.RegisteredBuffer(long(self.pos_vbo.buffers[0]), cudagl.graphics_map_flags.NONE)
        self.gpu_col = cudagl.RegisteredBuffer(long(self.col_vbo.buffers[0]), cudagl.graphics_map_flags.NONE)
        self.col_vbo.bind()
        self.gpu_vel = ga.to_gpu(ary=self.vel)

        self.pos_gen_cuda = cuda.mem_alloc(self.pos.nbytes)
        cuda.memcpy_htod(self.pos_gen_cuda, self.pos)
        self.vel_gen_cuda = cuda.mem_alloc(self.vel.nbytes)
        cuda.memcpy_htod(self.vel_gen_cuda, self.vel)

        self.ctx.synchronize()

    def load_gl_cuda_tex_buffer(self, im_init):
        self.cuda_pbo_resource = cudagl.BufferObject(int(self.pbo_im_buffer)) # Mapping GLBuffer to cuda_resource
        self.array = cuda.matrix_to_array(im_init, "C") # C-style instead of Fortran-style: row-major
        # self.array = ga.to_gpu(ary=im_init) # C-style instead of Fortran-style: row-major

        self.texref.set_array(self.array)
        self.texref.set_flags(cuda.TRSA_OVERRIDE_FORMAT)

    def modify_gl_cuda_buffer(self, id_fi, dt):
        # print "modify gl buffer"
        l_cont = np.copy(self.gpu_vect_frags.l_cont)
        n_l_cont_un = l_cont[l_cont == 1].shape[0]
        max_len = np.float32(l_cont.max())
        # print "max len = ", max_len
        self.id_contigs = np.copy(self.gpu_vect_frags.id_c)
        idc_un, idx_un = np.unique(self.id_contigs, return_index=True)
        n_new_contigs = len(idc_un)
        # idc_un = idc_un - idc_un.min()
        # print "n new contigs = ", n_new_contigs
        # print "id max = ", idc_un.max()
        # print "id min = ", idc_un.min()
        list_len_contigs = l_cont[idx_un]
        ord_length = np.argsort(list_len_contigs)
        # print "list lenght = ", list_len_contigs
        # print "ord length = ", ord_length
        # print "idc_un = ", idc_un
        # print "len idc_un = ", idc_un.shape
        # print "len ord_length = ", ord_length.shape

        # old_2_new_indexes = np.zeros((self.n_new_frags, ), dtype=np.int32)
        old_2_new_indexes = np.zeros((idc_un.max() + 1, ), dtype=np.int32)
        # print "len old_2_new_indexes = ", old_2_new_indexes.shape
        old_2_new_indexes[idc_un[ord_length]] = np.arange(0, n_new_contigs, 1, dtype=np.int32)
        # print "old_2 new indexes = ", old_2_new_indexes

        gpu_old_2_new = ga.to_gpu(old_2_new_indexes)
        # print sorted_length
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block) + 1, 1)
        map_pos = self.gpu_pos.map()
        (pos_ptr, pos_siz) = map_pos.device_ptr_and_size()
        map_col = self.gpu_col.map()
        (col_ptr, col_siz) = map_col.device_ptr_and_size()

        max_id = np.float32(n_new_contigs - 1)
        ##### update pos particles #####
        self.gl_update_pos(np.intp(pos_ptr),
                           np.intp(col_ptr),
                           self.gpu_vel,
                           self.pos_gen_cuda,
                           self.vel_gen_cuda,
                           self.gpu_vect_frags.get_ptr(),
                           gpu_old_2_new,
                           self.gpu_id_contigs,
                           max_len,
                           max_id,
                           self.n_new_frags,
                           np.int32(id_fi),
                           np.int32(n_l_cont_un),
                           self.rng_states, np.int32(self.n_generators,),
                           dt,
                           block=block_, grid=grid_)
        self.ctx.synchronize()
        map_pos.unmap()
        ##### update image #####
        size_block = (16, 16, 1)
        size_grid = (int(self.n_frags / 16) + 1, int(self.n_frags / 16) + 1)

        mapping_obj = self.cuda_pbo_resource.map()
        im_2_update = mapping_obj.device_ptr()

        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        pre_new_index = np.zeros((self.n_new_frags,), dtype=np.int32)
        vect_is_rep = np.zeros((self.n_new_frags,), dtype=np.int32)
        id_start_ctg = np.nonzero(c.start_bp == 0)[0]
        # print "id start ctg = ", id_start_ctg
        list_len_contigs = c.l_cont[id_start_ctg]
        list_len_contigs.sort()
        cum_sum = np.cumsum(list_len_contigs)
        to_gpu_cum_sum = [0]
        to_gpu_cum_sum.extend(list(cum_sum)[:-1])
        self.np_gpu_cum_sum = np.array(to_gpu_cum_sum, dtype=np.int32)
        for i in xrange(0, self.n_new_frags):
            pos = c.pos[i]
            id_c = c.id_c[i]
            id_d = c.id_d[i]
            is_rep = c.rep[i]
            pre_new_index[self.np_gpu_cum_sum[id_c] + pos] = (is_rep == 0) * id_d + (is_rep == 1) * -1
        new_index = pre_new_index[pre_new_index >= 0]
        # print "gpu cum sum = ", np_gpu_cum_sum
        # print "len(id_start_ctg) = ", len(id_start_ctg)

        gpu_cum_sum = ga.to_gpu(ary=self.np_gpu_cum_sum)
        gpu_new_index = ga.to_gpu(ary=new_index)
        self.gl_update_im(np.intp(im_2_update), gpu_new_index, np.int32(self.n_frags),
                          block=size_block, grid=size_grid, texrefs=[self.texref])
        self.ctx.synchronize()
        mapping_obj.unmap() # Unmap the GlBuffer
        max_id = np.int32(max_id)
        return max_id




    def step_max_likelihood(self, id_fA, delta, size_block, dt, t, n_step):


        if id_fA not in self.id_frags_blacklisted:
            start = cuda.Event()
            end = cuda.Event()
            tic = time.time()

            self.gpu_vect_frags.copy_from_gpu()
            id_start = np.nonzero(self.gpu_vect_frags.start_bp == 0)[0]
            if np.any(self.gpu_vect_frags.start_bp < 0):
                print "problem: negative distance !!!! @ frag: ", np.nonzero(self.gpu_vect_frags.start_bp < 0)
            # if np.any(self.gpu_vect_frags.prev[id_start] != -1):
            #     print "problem: id_prev not good !!!! @ frag: ", np.nonzero(self.gpu_vect_frags.prev[id_start] != -1)
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))

            mean_len = self.gpu_vect_frags.l_cont.mean()

            mean_len_bp = self.gpu_vect_frags.l_cont_bp[id_start].mean()

            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()

            self.curr_likelihood.fill(np.float64(0))
            self.gpu_vect_frags.copy_from_gpu()
            ###################################################
            block_ = (size_block, 1, 1)
            # stride = 10
            n_block = self.init_n_values_triu_extra // (size_block) + 1
            grid_all = (int(n_block / self.stride), 1)
            start.record()
            # print "dim block = ", block_
            # print "dim grid =", grid_all
            self.evaluate_likelihood(self.data,
                                     self.gpu_vect_frags.get_ptr(),
                                     self.gpu_collector_id_repeats,
                                     self.gpu_frag_dispatcher,
                                     self.gpu_sub_frag_id,
                                     self.gpu_rep_sub_frags_id,
                                     self.gpu_sub_frag_len_bp,
                                     self.gpu_sub_frag_accu,
                                     self.curr_likelihood,
                                     self.gpu_param_simu,
                                     self.init_n_values_triu,
                                     self.init_n_values_triu_extra,
                                     self.n_frags,
                                     self.init_n_sub_frags,
                                     self.mean_squared_frags_per_bin,
                                     block=block_, grid=grid_all, shared=0)
            end.record()
            end.synchronize()
            secs = start.time_till(end) * 1e-3
            # print "CUDA clock execution timing( likelihood full): ", secs
            likelihood_t = ga.sum(self.curr_likelihood, dtype=np.float64).get()
            # print "all likelihood computing time = ", time.time() - tic
            # print "current likelihood = ", likelihood_t
            self.likelihood_t = likelihood_t
            ###################################################
            tic_info = time.time()

            len_contig_A = self.gpu_vect_frags.l_cont[id_fA]

            contig_A = self.gpu_vect_frags.id_c[id_fA]
            # print "id_C contig_A = ", contig_A
            size_block_fill = 1024
            block_fill = (size_block_fill, 1, 1)
            grid_fill = (int(self.n_new_frags // size_block_fill + 1), 1)
            max_id = np.int32(ga.max(self.gpu_id_contigs).get())
            # print "max id = ", max_id
            ###################################################
            start.record()
            self.fill_sub_index_A(self.gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_A), self.n_new_frags,
                                  block=block_fill, grid=grid_fill)
            end.record()
            end.synchronize()
            # print "fill sub index execution time = ", time.time() - tic_info
            ###################################################
            # print id_fA
            id_neighbours = self.return_neighbours(id_fA, delta)
            # id_neighbours = self.old_return_neighbours(id_fA, delta)
            n_neighbours = len(id_neighbours)
            self.score = np.zeros((n_neighbours * self.n_tmp_struct,), dtype=np.float64)
            # print "pre score zeros = ", self.score
            ## TEST!!!!
            id_neighbours.sort() # maybe to reactivate!
            # id_neighbours.reverse() # 3d display in new_test_model
            ##########
            time_before_l = time.time()
            # print "time just before likelihood = ", time_before_l - tic
            # print "l cont fA = ", len_contig_A
            # print "n neighbors = ", n_neighbours
            for id_x in xrange(0, n_neighbours): # place where we want to spread the workload accross the network!
                self.gl_window.remote_update()
                id_fB = id_neighbours[id_x]
                # print "id_fB = ", id_fB
                self.stream_likelihood(id_fA, contig_A, len_contig_A, id_fB, id_x, likelihood_t, max_id)

            # print self.score
            # print "where = ", np.nonzero(self.score == -np.inf)
            # print "is nan = ", np.any(np.isnan(self.score))
            # print "min score = ", self.score.min()
            t_numpy = time.time()
            # print "score original = ", self.score
            ########################
            scores_2_remove = []
            scores_2_remove.extend(range(self.n_tmp_struct, len(self.score), self.n_tmp_struct)) # remove extra pop
            # scores_2_remove.extend(range(1, len(self.score), self.n_tmp_struct)) # remove all flip
            scores_2_remove.extend(range(self.n_tmp_struct + 1, len(self.score), self.n_tmp_struct)) # remove extra flip

            # for id_modif in xrange(12, self.n_tmp_struct):
            #     scores_2_remove.extend(range(self.n_tmp_struct + id_modif, len(self.score), self.n_tmp_struct)) # remove extra local flip
            # print "scores 2 remove = ", scores_2_remove
            ########################
            # print "score = ", self.score
            id_max = self.score.argmax()
            or_score = np.copy(self.score)
            filtered_score = self.score - self.score.min()
            ########################
            filtered_score[scores_2_remove] = 0
            # print "filtered score ( before) = ", filtered_score
            ########################
            max_score = filtered_score.max()
            thresh_overflow = 30
            # filtered_score[filtered_score < max_score - thresh] = 0
            filtered_score = filtered_score - (max_score - thresh_overflow)
            filtered_score[filtered_score < 0] = 0
            # print "filtered score (after)= ", filtered_score
            # id_ok_4_sampling = np.ix_(filtered_score >= max_score - thresh)
            id_ok_4_sampling = np.ix_(filtered_score > 0)
            # print "id ok for sampling = ", id_neighbours[id_ok_4_sampling[0] / self.n_tmp_struct]
            self.sub_score = filtered_score[id_ok_4_sampling]
            # print "sub score (before)= ", sub_score
            ############# DEBUGGGG ###########################
            # sub_score = np.exp(sub_score) ## debuggg !!!!
            ############# DEBUGGGG ###########################
            # print "sub score = ", sub_score
            ############# DEBUGGGG ###########################
            F_t = self.temperature(t, n_step)
            self.sub_score = self.sub_score / self.sub_score.sum()
            self.sub_score[self.sub_score > 0] = np.power(self.sub_score[self.sub_score > 0], 1./F_t)
            ############# DEBUGGGG ###########################
            self.sub_score = self.sub_score / self.sub_score.sum()
            tic_sampling = time.time()
            ### DEBUGGG #######
            if len(id_ok_4_sampling[0]) == 1 or len(id_ok_4_sampling[0]) == 0:
                sample_out = id_max
            else:
                sample_out = np.random.choice(id_ok_4_sampling[0], 1, p=self.sub_score)[0]
            ### DEBUGGG #######
            # sample_out = id_max
            ### DEBUGGG #######
            id_f_sampled = id_neighbours[sample_out / self.n_tmp_struct]
            op_sampled = sample_out % self.n_tmp_struct
            # print "id frag sampled = ", id_f_sampled
            # print "operation sampled = ", self.modification_str[op_sampled]
            # print 'id operation =', op_sampled

            self.test_copy_struct(id_fA, id_f_sampled, op_sampled, max_id)

            # print or_score
            # print sample_out
            tac = time.time()
            # print "numpy bottle time execution = ", time.time() - t_numpy
            # print "copy struct execution time = ", tac - tic_copy_struct
            # print "execution time (all)= ", tac - tic
            o = or_score[sample_out]
            self.o = o
        else:
            print "blacklist frag"
            o = self.o
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)
            id_start = np.nonzero(self.gpu_vect_frags.start_bp == 0)[0]

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))

            mean_len = self.gpu_vect_frags.l_cont.mean()
            mean_len_bp = self.gpu_vect_frags.l_cont_bp[id_start].mean()
            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()
            op_sampled = -1
            id_f_sampled = id_fA
            F_t = self.temperature(t, n_step)
        dist = self.dist_inter_genome(self.gpu_vect_frags)
        self.likelihood_t = o
        return o, n_contigs, min_len, mean_len_bp, max_len, op_sampled, id_f_sampled, dist, F_t

    def return_rippe_vals(self, p0):
        y_eval = opti.peval(self.bins, p0)
        return y_eval

    def compute_likelihood_4_nuisance(self,):
        start = cuda.Event()
        end = cuda.Event()
        size_block = 512
        block_ = (size_block, 1, 1)
        # stride = 10
        n_block = self.init_n_values_triu_extra // (size_block) + 1
        grid_all = (int(n_block / self.stride), 1)
        start.record()
        # print "dim block = ", block_
        # print "dim grid =", grid_all
        self.evaluate_likelihood(self.data,
                                 self.gpu_vect_frags.get_ptr(),
                                 self.gpu_collector_id_repeats,
                                 self.gpu_frag_dispatcher,
                                 self.gpu_sub_frag_id,
                                 self.gpu_rep_sub_frags_id,
                                 self.gpu_sub_frag_len_bp,
                                 self.gpu_sub_frag_accu,
                                 self.curr_likelihood,
                                 self.gpu_param_simu_test,
                                 self.init_n_values_triu,
                                 self.init_n_values_triu_extra,
                                 self.n_frags,
                                 self.init_n_sub_frags,
                                 self.mean_squared_frags_per_bin,
                                 block=block_, grid=grid_all, shared=0)
        end.record()
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        # print "CUDA clock execution timing( likelihood full): ", secs
        likelihood_star = ga.sum(self.curr_likelihood, dtype=np.float64).get()

        return likelihood_star


    def step_nuisance_parameters(self, dt, t, n_step):


        self.gpu_vect_frags.copy_from_gpu()
        self.gl_window.remote_update()

        curr_param = np.copy(self.param_simu)
        kuhn, lm, c1, slope, d, d_max, fact, d_nuc = curr_param[0]

        self.range_scale_fact = [0, self.hic_matrix.max()]
        self.sigma_fact = 10**(np.log10(fact)-2)  # for G1

        self.range_slope = [-0.5, -2]
        self.sigma_slope = 0.05 # ok all

        self.range_d_max = [0, 10000]
        # self.sigma_d_max = 100 # ok all
        self.sigma_d_max = 100 # ok all

        self.range_d_nuc = [0, 100]
        self.sigma_d_nuc = 0.5 # test s1

        self.range_d = [0, 200]
        self.sigma_d = 10 # ok for s1

        # randomly select a modifier
        id_modif = np.random.choice(4)

        if id_modif == 0: # scale factor
            new_fact = fact + np.random.normal(loc=0.0, scale=self.sigma_fact)
            test_param = [kuhn, lm, slope, d, new_fact]
            new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
            out_test_param = [(kuhn, lm, c1, slope, d, new_d_max, new_fact, d_nuc)]
        elif id_modif == 1: # sclope
            new_slope = slope + np.random.normal(loc=0.0, scale=self.sigma_slope)
            test_param = [kuhn, lm, new_slope, d, fact]
            new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            c1 = np.float32((0.53 * np.power(lm / kuhn, new_slope)) * np.power(kuhn, -3))
            out_test_param = [(kuhn, lm, c1, new_slope, d, new_d_max, fact, d_nuc)]
        elif id_modif == 2:# max distance intra
            new_d_max = d_max + np.random.normal(loc=0.0, scale=self.sigma_d_max)
            test_param = [kuhn, lm, slope, d, fact]
            # new_d_nuc = opti1.peval(new_d_max, test_param)
            new_d_nuc = opti.peval(new_d_max, test_param)
            c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
            out_test_param = [(kuhn, lm, c1, slope, d, new_d_max, fact, new_d_nuc)]
        elif id_modif == 3: # val trans
            new_d_nuc = d_nuc + np.random.normal(loc=0.0, scale=self.sigma_d_nuc)
            test_param = [kuhn, lm, slope, d, fact]
            new_d_max = opti.estimate_max_dist_intra(test_param, new_d_nuc)
            c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
            out_test_param = [(kuhn, lm, c1, slope, d, new_d_max, fact, new_d_nuc)]
        else: # d
            new_d = d + np.random.normal(loc=0.0, scale=self.sigma_d)
            # new_d = min(new_d, self.range_d[1])
            # new_d = max(new_d, self.range_d[0])

            test_param = [kuhn, lm, slope, new_d, fact]
            new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            c1 = np.float32((0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3))
            out_test_param = [(kuhn, lm, c1, slope, new_d, new_d_max, fact, d_nuc)]

        out_test_param = np.array(out_test_param, dtype=self.param_simu_T)
        #print "test param = ", test_param
        #print "out test param = ",out_test_param
        cuda.memcpy_htod(self.gpu_param_simu_test, out_test_param)

        test_likelihood = self.compute_likelihood_4_nuisance()
        F_t = self.temperature(t, n_step)
        ratio = np.exp((test_likelihood - self.likelihood_t) / F_t)
        u = np.random.rand()
        success = 0
        if ratio >= u:
            # if id_modif == 2:
            #     print "success =====> ", ratio
            #     print "d_max = ", new_d_max
            success = 1
            cuda.memcpy_htod(self.gpu_param_simu, out_test_param)
            self.param_simu = out_test_param
            self.likelihood_t = test_likelihood
            # print "param simu = ", self.param_simu
        kuhn, lm, c1, slope, d, d_max, fact, d_nuc = self.param_simu[0]
        p0 = [kuhn, lm, slope, d, fact]
        y_rippe = self.return_rippe_vals(p0)
        return fact, d, d_max, d_nuc, slope, self.likelihood_t, success, y_rippe

    def debug_step_max_likelihood(self, id_fA, delta, size_block, dt):


        if id_fA not in self.id_frags_blacklisted:
            start = cuda.Event()
            end = cuda.Event()
            tic = time.time()

            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))

            mean_len = self.gpu_vect_frags.l_cont.mean()
            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()

            self.curr_likelihood.fill(np.float64(0))
            self.gpu_vect_frags.copy_from_gpu()
            ###################################################
            block_ = (size_block, 1, 1)
            # stride = 10
            n_block = self.init_n_values_triu_extra // (size_block) + 1
            grid_all = (int(n_block / self.stride), 1)
            start.record()
            # print "dim block = ", block_
            # print "dim grid =", grid_all
            self.evaluate_likelihood(self.data,
                                     self.gpu_vect_frags.get_ptr(),
                                     self.gpu_collector_id_repeats,
                                     self.gpu_frag_dispatcher,
                                     self.gpu_sub_frag_id,
                                     self.gpu_rep_sub_frags_id,
                                     self.gpu_sub_frag_len_bp,
                                     self.gpu_sub_frag_accu,
                                     # self.gpu_full_expected_lin,
                                     # self.gpu_full_expected_rep_lin,
                                     self.curr_likelihood,
                                     self.gpu_param_simu,
                                     self.init_n_values_triu,
                                     # self.init_n_values_triu,
                                     self.init_n_values_triu_extra,
                                     self.n_frags,
                                     self.init_n_sub_frags,
                                     self.mean_squared_frags_per_bin,
                                     block=block_, grid=grid_all, shared=0)
            end.record()
            end.synchronize()
            secs = start.time_till(end) * 1e-3
            # print "CUDA clock execution timing( likelihood full): ", secs
            likelihood_t = ga.sum(self.curr_likelihood, dtype=np.float64).get()
            # print "all likelihood computing time = ", time.time() - tic
            # print "current likelihood = ", likelihood_t
            self.likelihood_t = likelihood_t
            ###################################################
            tic_info = time.time()

            len_contig_A = self.gpu_vect_frags.l_cont[id_fA]

            contig_A = self.gpu_vect_frags.id_c[id_fA]
            # print "id_C contig_A = ", contig_A
            size_block_fill = 1024
            block_fill = (size_block_fill, 1, 1)
            grid_fill = (int(self.n_new_frags // size_block_fill + 1), 1)
            max_id = np.int32(ga.max(self.gpu_id_contigs).get())
            # print "max id = ", max_id
            ###################################################
            start.record()
            self.fill_sub_index_A(self.gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_A), self.n_new_frags,
                                  block=block_fill, grid=grid_fill)
            end.record()
            end.synchronize()
            # print "fill sub index execution time = ", time.time() - tic_info
            ###################################################

            id_neighbours = self.return_neighbours(id_fA, delta)
            n_neighbours = len(id_neighbours)
            self.score = np.zeros((n_neighbours * self.n_tmp_struct,), dtype=np.float32)
            id_neighbours.sort()
            time_before_l = time.time()
            # print "time just before likelihood = ", time_before_l - tic
            # print "l cont fA = ", len_contig_A
            for id_x in xrange(0, n_neighbours):
                # print "id frag =", id_x
                self.gl_window.remote_update()
                id_fB = id_neighbours[id_x]
                # print "id_fB = ", id_fB
                self.new_perform_modificationS(id_fA, id_fB, max_id, True)
                for id_mode in xrange(0, self.n_tmp_struct):
                    self.evaluate_likelihood(self.data,
                                             self.collector_gpu_vect_frags[id_mode].get_ptr(),
                                             self.gpu_collector_id_repeats,
                                             self.gpu_frag_dispatcher,
                                             self.gpu_sub_frag_id,
                                             self.gpu_rep_sub_frags_id,
                                             self.gpu_sub_frag_len_bp,
                                             self.gpu_sub_frag_accu,
                                             # self.gpu_full_expected_lin,
                                             # self.gpu_full_expected_rep_lin,
                                             self.curr_likelihood_forward,
                                             self.gpu_param_simu,
                                             self.init_n_values_triu,
                                             # self.init_n_values_triu,
                                             self.init_n_values_triu_extra,
                                             self.n_frags,
                                             self.init_n_sub_frags,
                                             self.mean_squared_frags_per_bin,
                                             block=block_, grid=grid_all, shared=0)
                    end.record()
                    end.synchronize()
                    likelihood_tmp = ga.sum(self.curr_likelihood_forward, dtype=np.float64).get()
                    self.score[id_x * self.n_tmp_struct + id_mode] = likelihood_tmp

            # print self.score
            # print "where = ", np.nonzero(self.score == -np.inf)
            # print "is nan = ", np.any(np.isnan(self.score))
            # print "min score = ", self.score.min()
            t_numpy = time.time()
            # print "score original = ", self.score
            ########################
            scores_2_remove = []
            scores_2_remove.extend(range(self.n_tmp_struct, len(self.score), self.n_tmp_struct)) # remove extra pop
            # scores_2_remove.extend(range(1, len(self.score), self.n_tmp_struct)) # remove all flip
            scores_2_remove.extend(range(self.n_tmp_struct + 1, len(self.score), self.n_tmp_struct)) # remove extra flip
            # print "scores 2 remove = ", scores_2_remove
            ########################
            id_max = self.score.argmax()
            or_score = np.copy(self.score)
            filtered_score = self.score - self.score.min()
            ########################
            filtered_score[scores_2_remove] = 0
            # print "filtered score ( before) = ", filtered_score
            ########################
            max_score = filtered_score.max()
            thresh_overflow = 600
            # filtered_score[filtered_score < max_score - thresh] = 0
            filtered_score = filtered_score - (max_score - thresh_overflow)
            filtered_score[filtered_score < 0] = 0
            # print "filtered score (after)= ", filtered_score
            # id_ok_4_sampling = np.ix_(filtered_score >= max_score - thresh)
            id_ok_4_sampling = np.ix_(filtered_score > 0)
            # print "id ok for sampling = ", id_neighbours[id_ok_4_sampling[0] / self.n_tmp_struct]
            sub_score = filtered_score[id_ok_4_sampling]
            # print "sub score (before)= ", sub_score
            # sub_score = np.exp(sub_score - sub_score.min())
            # print "sub score = ", sub_score
            sub_score = sub_score / sub_score.sum()
            tic_sampling = time.time()
            if len(id_ok_4_sampling[0]) == 1:
                sample_out = id_max
            else:
                sample_out = np.random.choice(id_ok_4_sampling[0], 1, p=sub_score)[0]

            id_f_sampled = id_neighbours[sample_out / self.n_tmp_struct]
            op_sampled = sample_out % self.n_tmp_struct
            # print "id frag sampled = ", id_f_sampled
            # print "operation sampled = ", self.modification_str[op_sampled]
            # print 'id operation =', op_sampled

            self.test_copy_struct(id_fA, id_f_sampled, op_sampled, max_id)

            # print or_score
            # print sample_out
            tac = time.time()
            # print "numpy bottle time execution = ", time.time() - t_numpy
            # print "copy struct execution time = ", tac - tic_copy_struct
            # print "execution time (all)= ", tac - tic
            o = or_score[sample_out]
            self.o = o
        else:
            print "blacklist frag"
            o = self.o
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))

            mean_len = self.gpu_vect_frags.l_cont.mean()
            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()
            op_sampled = -1
            id_f_sampled = id_fA

        return o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled


    def return_neighbours(self, id_fA, delta0):
        # print "id_frag = ", id_fA
        ori_id = self.gpu_vect_frags.id_d[id_fA]
        delta = min(self.n_neighbors, delta0)


        # DEBUG
        # if ori_id in self.id_frag_duplicated:
        #     delta = max(self.n_neighbors, delta)

        # fact = 3
        # pk = self.distri_frags[ori_id]['pk']**fact
        # distri = pk / pk.sum()

        distri = self.distri_frags[ori_id]['pk']
        n_max_candidates = min(delta, np.nonzero(distri != 0)[0].shape[0])

        init_id = np.random.choice(self.distri_frags[ori_id]['xk'], n_max_candidates, p=distri,
                                   replace=False)
        out = []

        if ori_id in self.id_frag_duplicated:
            d = self.frag_dispatcher[ori_id]
            l = self.collector_id_repeats[d['x']: d['y']]
            dup = np.lib.arraysetops.setdiff1d(l, id_fA)
            out.extend(dup)

        for id_fB in init_id:
            d = self.frag_dispatcher[id_fB]
            out.extend(self.collector_id_repeats[d['x']: d['y']])

        real_out = []
        for ele in out:
            if ele not in self.id_frags_blacklisted:
                real_out.append(ele)

        return real_out

    def old_return_neighbours(self, id_fA, delta):
        ori_id = self.gpu_vect_frags.id_d[id_fA]
        if ori_id in self.id_frag_duplicated:
            delta = delta * 15
        init_id = np.copy(self.sorted_neighbours[ori_id, -delta:])
        out = []

        if ori_id in self.id_frag_duplicated:
            d = self.frag_dispatcher[ori_id]
            l = self.collector_id_repeats[d['x']: d['y']]
            dup = np.lib.arraysetops.setdiff1d(l, id_fA)
            out.extend(dup)

        for id_fB in init_id:
            d = self.frag_dispatcher[id_fB]
            out.extend(self.collector_id_repeats[d['x']: d['y']])

        # for id_fB in init_id:
        #     if id_fB in self.id_frag_duplicated:
        #         d = self.frag_dispatcher[id_fB]
        #         out.extend(self.collector_id_repeats[d['x']: d['y']])
        #     else:
        #         out.append(id_fB)
        real_out = []
        for ele in out:
            if ele not in self.id_frags_blacklisted:
                real_out.append(ele)
        return real_out


    def setup_distri_frags(self,):
        # generates random variables for every frags
        self.distri_frags = dict()
        fact = 3
        for i in range(0, self.n_frags):
            v = np.float32(self.hic_matrix_sub_sampled[i, :])
            vtmp = np.copy(v)
            # xk = np.nonzero(vtmp > vtmp.mean() + vtmp.std() * 3)[0]
            id_sort = np.argsort(vtmp)
            id_sort_l = list(id_sort)
            id_sort_l.reverse()
            id_sort_l = np.array(id_sort_l, dtype=np.int32)
            xk = id_sort_l[: self.n_neighbors]

            dat = vtmp[xk]**fact

            if dat.sum() >0:
                pk = dat / dat.sum()
            else:
                tmp = np.ones_like(dat, dtype=np.float32)
                pk = tmp / tmp.sum()

            # pk = vtmp[xk] / vtmp[xk].sum()

            self.distri_frags[i] = dict()
            self.distri_frags[i]['distri'] = stats.rv_discrete(name='frag_'+str(i), values=(xk, pk))
            self.distri_frags[i]['xk'] = xk
            self.distri_frags[i]['pk'] = pk

    def stream_likelihood(self, id_fA, contig_A, len_contig_A, id_fB, id_x, likelihood_t, max_id):
        tic_start = time.time()
        start = cuda.Event()
        end = cuda.Event()
        stream, event = [], []
        marker_names = ['start', 'post_likelihood', 'post_reduction', 'end']
        for j in xrange(0, self.n_tmp_struct):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))
        size_block = 512
        ###################################################
        len_contig_B = self.gpu_vect_frags.l_cont[id_fB]
        contig_B = self.gpu_vect_frags.id_c[id_fB]
        # print "contig B = ", contig_B
        block_fill = (size_block, 1, 1)
        grid_fill = (int(self.n_new_frags // size_block + 1), 1)
        self.new_perform_modificationS(id_fA, id_fB, max_id, id_x == 0)
        ### DEBUG start ###
        # for kl in range(0, self.n_tmp_struct):
        #     ele = self.collector_gpu_vect_frags[kl]
            # self.diagnosis(ele, id_fA, id_fB, kl)
            # ele.copy_from_gpu()
            # id_start = np.nonzero(ele.start_bp == 0)[0]
            # print "shape id start ", id_start.shape
            # t1 = ele.start_bp == 0
            # t2 = ele.prev != -1
            # t3 = ele.circ == 0
            # test = t1 * t2 * t3
            # if np.any(ele.start_bp < 0):
            #     id_problem = np.nonzero(ele.start_bp < 0)[0]
            #     print "problem START problem problem @ ", id_problem
            #     print "id_fA = ", id_fA
            #     print "id_fB = ", id_fB
            #     print "mutation = ", kl
            #     print "is circular = ", ele.circ[id_problem]
            #
            # if np.any(test):
            #     id_problem = np.nonzero(test)[0]
            #     print "n troubles = ", id_problem.shape
            #     print "problem PREV problem problem @ ", id_problem
            #     print "id_fA = ", id_fA
            #     print "id_fB = ", id_fB
            #     print "mutation = ", kl
            #     print "is circular = ", ele.circ[id_problem]
            #     raw_input(" SO ???")

        ### DEBUG end ####
        ###################################################

        if contig_B != contig_A:
            start.record()
            self.fill_sub_index_B(self.gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_B),
                                  np.int32(len_contig_A),
                                  self.n_new_frags,
                                  block=block_fill, grid=grid_fill)
            end.record()
            end.synchronize()
            size_sub_index = len_contig_A + len_contig_B
        else:
            size_sub_index = len_contig_A
        ###################################################
        init_sub_index_tmp = self.gpu_sub_index.get()
        init_sub_index = init_sub_index_tmp[:size_sub_index]


        sub_index_no_repeats = np.lib.arraysetops.setdiff1d(init_sub_index, self.np_id_frag_duplicated)
        sub_index_repeats = np.lib.arraysetops.intersect1d(init_sub_index, self.np_id_frag_duplicated)
        if sub_index_repeats.shape[0] == 0:
            sub_index_repeats = np.array([-1], dtype=np.int32)
            size_sub_index_repeats = np.int32(0)
        else:
            # print "sub index repeats = ", sub_index_repeats
            size_sub_index_repeats = sub_index_repeats.shape[0]

        if sub_index_no_repeats.shape[0] == 0:
            sub_index_no_repeats = np.array([-1], dtype=np.int32)
            size_sub_index_no_repeats = np.int32(0)
        else:
            size_sub_index_no_repeats = sub_index_no_repeats.shape[0]

        # print "sub_index no repeats = ", sub_index_no_repeats
        #
        # print "sub_index repeats = ", sub_index_repeats

        # size_sub_index_no_repeats = sub_index_no_repeats.shape[0]
        n_repeats = np.int32(size_sub_index_repeats)
        n_vals_intra_repeats = np.int32(n_repeats * (n_repeats - 1) / 2)

        n_vals_to_update_no_repeats = np.int32(size_sub_index_no_repeats * (size_sub_index_no_repeats - 1) / 2)
        lim_repeats_vs_uniq = np.int32(n_vals_to_update_no_repeats + n_repeats * self.n_frags_uniq)
        lim_intra_repeats = lim_repeats_vs_uniq + n_vals_intra_repeats
        n_values_to_update = lim_intra_repeats + n_repeats

        # print "n vals no repeats = ", n_vals_to_update_no_repeats
        # print "lim repeats vs uniq = ", lim_repeats_vs_uniq
        # print "lim intra repeats = ", lim_intra_repeats
        # print "n values to update = ", n_values_to_update
        # print "n_repeats = ", n_repeats
        # print "sub_ index repeats = ", sub_index_repeats
        # print "sub_ index no repeats = ", sub_index_no_repeats

        gpu_sub_index_no_repeats = ga.to_gpu(ary=sub_index_no_repeats)
        gpu_list_repeats = ga.to_gpu(ary=sub_index_repeats)

        dim_grid = int(n_values_to_update)
        block_ = (size_block, 1, 1)
        n_block = int(dim_grid // size_block + 1)
        grid_ = (n_block // self.stride + 1, 1)
        # print "grid = ", grid_
        # print "block = ", block_
        size_shared = size_block * 8
        gpu_tmp_likelihood = []
        cpu_tmp_likelihood = []
        for i in xrange(0, self.n_tmp_struct):
            gpu_tmp_likelihood.append(ga.to_gpu(np.zeros((1,), dtype=np.float64)))
            cpu_tmp_likelihood.append(np.float64(0))
        for j in xrange(0, self.n_tmp_struct):
            event[j]['start'].record(stream[j])
            self.sub_compute_likelihood_1d(self.data,
                                           self.collector_gpu_vect_frags[j].get_ptr(),
                                           gpu_sub_index_no_repeats,
                                           gpu_list_repeats,
                                           self.gpu_uniq_frags,
                                           self.gpu_collector_id_repeats,
                                           self.gpu_frag_dispatcher,
                                           self.gpu_sub_frag_id,
                                           self.gpu_sub_frag_len_bp,
                                           self.gpu_sub_frag_accu,
                                           gpu_tmp_likelihood[j],
                                           self.curr_likelihood,
                                           self.gpu_param_simu,
                                           n_vals_to_update_no_repeats,
                                           lim_repeats_vs_uniq,
                                           lim_intra_repeats,
                                           n_values_to_update,
                                           self.n_frags_uniq,
                                           n_repeats,
                                           self.init_n_sub_frags,
                                           self.n_frags,
                                           self.mean_squared_frags_per_bin,
                                           # block=block_, grid=grid_, stream=stream[j])
                                           block=block_, grid=grid_, shared=size_shared, stream=stream[j])

        for j in xrange(0, self.n_tmp_struct): # Commenting out this line should break concurrency.
            event[j]['end'].record(stream[j])
            stream[j].synchronize()
            cpu_tmp_likelihood[j] = gpu_tmp_likelihood[j].get()

            # print "single stream execution time  =" , event[j]['start'].time_till(event[j]['end']) * 1e-3
            # print "likelihood execution time = ", event[j]['start'].time_till(event[j]['post_likelihood']) * 1e-3
            # print "reduction execution time = ", event[j]['post_likelihood'].time_till(event[j]['post_reduction']) * 1e-3

        for j in xrange(0, self.n_tmp_struct):
            self.score[id_x * self.n_tmp_struct + j] = cpu_tmp_likelihood[j] + likelihood_t
        # print "execution time all streams = ",time.time() - tic_start

    def define_neighbourhood(self,):

        mat_norm = np.array(self.norm_vect_accu.T * self.norm_vect_accu, dtype=np.float32)
        self.matrix_normalized = self.hic_matrix_sub_sampled / mat_norm
        tmp_sorted = self.matrix_normalized.argsort(axis=1)

        sorted_neighbours = []
        for i in range(0, int(self.n_frags)):
            all_idx = tmp_sorted[i, :]
            line = list(all_idx)
            pos = np.nonzero(np.array(line) == i)[0][0]
            line.pop(pos)
            sorted_neighbours.append(np.array(line, dtype=np.int32))
        self.sorted_neighbours = np.array(sorted_neighbours, dtype=np.int32)

    def set_jumping_distributions_parameters(self, delta):
        self.define_neighbourhood()
        self.jump_dictionnary = dict()
        for i in xrange(0, self.n_frags):
            id_neighbours = self.sorted_neighbours[i, -delta:]
            id_no_neighbours = self.sorted_neighbours[i, 0:delta]
            scores = np.array(self.matrix_normalized[i, id_neighbours], dtype=np.float32)
            val_mean = self.matrix_normalized[i, id_no_neighbours].mean()
            norm_scores = scores / scores.sum()
            self.jump_dictionnary[i] = dict()
            self.jump_dictionnary[i]['proba'] = norm_scores
            self.jump_dictionnary[i]['frags'] = np.array(id_neighbours, dtype=np.int32)
            self.jump_dictionnary[i]['distri'] = np.zeros((self.n_frags), dtype=np.float32)
            self.jump_dictionnary[i]["set_frags"] = set()
            for k in range(0, delta):
                id_frag = self.jump_dictionnary[i]['frags'][k]
                self.jump_dictionnary[i]['set_frags'].add(id_frag)
                proba = self.jump_dictionnary[i]['proba'][k]
                self.jump_dictionnary[i]['distri'][id_frag] = proba
        # test = True
        # while test:
        #     id = raw_input(" give a frag id ?")
        #     id = int(id)
        #     print self.jump_dictionnary[id]
        #     test = raw_input(" keep on ? ")
        #     test = int(test) != 0

    def temperature(self, t, n_step):
        # T0 = np.float32(6 * 10 ** 3)
        # Tf = np.float32(6*10 ** 2)
        #
        # n_step = n_step
        # limit_rejection = 0.5
        # if t <= n_step * limit_rejection:
        #     val = T0 * (Tf / T0)**(t / (n_step* limit_rejection))
        # else:
        #     val = T0 * (Tf / T0)**(limit_rejection)
        #     # val = Tf
        # # print "temperature = ", val
        val = 1.0
        return val

    def free_gpu(self,):

        self.gpu_vect_frags.__del__()
        self.rng_states.free()
        (free, total) = cuda.mem_get_info()
        print("Global memory occupancy after cleaning processes: %f%% free"%(free*100/total))
        print("Global free memory  :%i Mo free"%(free/10**6))
        self.ctx.detach()
        del self.module

    def compute_all_score_MH(self, id_fA, V_set, forward):

        list_fB = list(V_set)
        n_neighbours = len(list_fB)
        score = np.zeros((self.n_modif_metropolis * n_neighbours), dtype=np.float64)
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
            vect_likelihood = self.curr_likelihood
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward
            vect_likelihood = self.curr_likelihood_forward

        gpu_vect_frags.copy_from_gpu()
        likelihood_t = self.compute_likelihood(gpu_vect_frags, vect_likelihood)
        start = cuda.Event()
        end = cuda.Event()
        max_id = gpu_vect_frags.id_c.max()
        contig_A = gpu_vect_frags.id_c[id_fA]
        len_contig_A = gpu_vect_frags.l_cont[id_fA]
        size_block_fill = 1024
        block_fill = (size_block_fill, 1, 1)
        grid_fill = (int(self.n_frags // size_block_fill + 1), 1)
        ###################################################
        start.record()
        self.fill_sub_index_A(gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_A), self.n_frags,
                              block=block_fill, grid=grid_fill)
        end.record()
        end.synchronize()
        ###################################################
        for id_x in xrange(0, n_neighbours):
            id_fB = list_fB[id_x]
            self.multi_likelihood_4_metropolis(id_fA, contig_A, len_contig_A, id_fB, id_x, gpu_vect_frags,
                                               likelihood_t, vect_likelihood, max_id, score, forward)

        return score

    def all_modifications_metropolis(self, id_fA, id_fB, max_id, forward):

        for i in xrange(0, 6):
            self.pop_out_pop_in_4_mh(id_fA, id_fB, i, max_id, forward)
        self.split_4_mh(id_fA, max_id, forward)
        self.paste_4_mh(id_fA, id_fB, max_id, forward)
        self.transloc_4_mh(id_fA, id_fB, max_id, forward)

    def multi_likelihood_4_metropolis(self, id_fA, contig_A, len_contig_A, id_fB, id_x,
                                      gpu_vect_frags, likelihood_t, likelihood_vect,
                                      max_id, score, forward):
        tic_start = time.time()
        start = cuda.Event()
        end = cuda.Event()
        stream, event = [], []
        marker_names = ['start', 'post_likelihood', 'post_reduction', 'end']
        for j in xrange(0, self.n_modif_metropolis):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))
        size_block = 512
        ###################################################
        len_contig_B = gpu_vect_frags.l_cont[id_fB]
        contig_B = gpu_vect_frags.id_c[id_fB]
        # print "contig B = ", contig_B
        block_fill = (size_block, 1, 1)
        grid_fill = (int(self.n_frags // size_block + 1), 1)
        self.all_modifications_metropolis(id_fA, id_fB, max_id, forward)
        ###################################################
        if contig_B != contig_A:
            start.record()
            self.fill_sub_index_B(gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_B),
                                  np.int32(len_contig_A),
                                  self.n_frags,
                                  block=block_fill, grid=grid_fill)
            end.record()
            end.synchronize()
            size_sub_index = len_contig_A + len_contig_B
        else:
            size_sub_index = len_contig_A
        ###################################################
        # dim_grid = size_sub_index * (size_sub_index - 1) / 2
        # block_ = (size_block, 1, 1)
        # n_block = int(dim_grid // size_block + 1)
        # grid_ = (n_block // 32 + 1, 1)
        # # print "grid = ", grid_
        # size_shared = size_block * 8
        # gpu_tmp_likelihood = []
        # cpu_tmp_likelihood = []

        init_sub_index_tmp = self.gpu_sub_index.get()
        init_sub_index = init_sub_index_tmp[:size_sub_index]


        sub_index_no_repeats = np.lib.arraysetops.setdiff1d(init_sub_index, self.np_id_frag_duplicated)
        sub_index_repeats = np.lib.arraysetops.intersect1d(init_sub_index, self.np_id_frag_duplicated)
        if sub_index_repeats.shape[0] == 0:
            sub_index_repeats = np.array([-1], dtype=np.int32)
            size_sub_index_repeats = np.int32(0)
        else:
            # print "sub index repeats = ", sub_index_repeats
            size_sub_index_repeats = sub_index_repeats.shape[0]

        if sub_index_no_repeats.shape[0] == 0:
            sub_index_no_repeats = np.array([-1], dtype=np.int32)
            size_sub_index_no_repeats = np.int32(0)
        else:
            size_sub_index_no_repeats = sub_index_no_repeats.shape[0]

        # print "sub_index no repeats = ", sub_index_no_repeats
        #
        # print "sub_index repeats = ", sub_index_repeats

        # size_sub_index_no_repeats = sub_index_no_repeats.shape[0]
        n_repeats = np.int32(size_sub_index_repeats)
        n_vals_intra_repeats = np.int32(n_repeats * (n_repeats - 1) / 2)

        n_vals_to_update_no_repeats = np.int32(size_sub_index_no_repeats * (size_sub_index_no_repeats - 1) / 2)
        lim_repeats_vs_uniq = np.int32(n_vals_to_update_no_repeats + n_repeats * self.n_frags_uniq)
        lim_intra_repeats = lim_repeats_vs_uniq + n_vals_intra_repeats
        n_values_to_update = lim_intra_repeats + n_repeats

        # print "n vals no repeats = ", n_vals_to_update_no_repeats
        # print "lim repeats vs uniq = ", lim_repeats_vs_uniq
        # print "lim intra repeats = ", lim_intra_repeats
        # print "n values to update = ", n_values_to_update
        # print "n_repeats = ", n_repeats
        # print "sub_ index repeats = ", sub_index_repeats
        # print "sub_ index no repeats = ", sub_index_no_repeats

        gpu_sub_index_no_repeats = ga.to_gpu(ary=sub_index_no_repeats)
        gpu_list_repeats = ga.to_gpu(ary=sub_index_repeats)

        dim_grid = int(n_values_to_update)
        block_ = (size_block, 1, 1)
        n_block = int(dim_grid // size_block + 1)
        grid_ = (n_block // self.stride + 1, 1)
        # print "grid = ", grid_
        # print "block = ", block_
        size_shared = size_block * 8
        gpu_tmp_likelihood = []
        cpu_tmp_likelihood = []



        for i in xrange(0, self.n_modif_metropolis):
            gpu_tmp_likelihood.append(ga.to_gpu(np.zeros((1,), dtype=np.float64)))
            cpu_tmp_likelihood.append(np.float64(0))

        for j in xrange(0, self.n_modif_metropolis):
            event[j]['start'].record(stream[j])
            mode = j
            self.sub_compute_likelihood_1d(self.data,
                                           self.collector_gpu_vect_frags[j].get_ptr(),
                                           gpu_sub_index_no_repeats,
                                           gpu_list_repeats,
                                           self.gpu_uniq_frags,
                                           self.gpu_collector_id_repeats,
                                           self.gpu_frag_dispatcher,
                                           self.gpu_sub_frag_id,
                                           self.gpu_sub_frag_len_bp,
                                           self.gpu_sub_frag_accu,
                                           gpu_tmp_likelihood[j],
                                           likelihood_vect,
                                           self.gpu_param_simu,
                                           n_vals_to_update_no_repeats,
                                           lim_repeats_vs_uniq,
                                           lim_intra_repeats,
                                           n_values_to_update,
                                           self.n_frags_uniq,
                                           n_repeats,
                                           self.init_n_sub_frags,
                                           self.n_frags,
                                           self.mean_squared_frags_per_bin,
                                           # block=block_, grid=grid_, stream=stream[j])
                                           block=block_, grid=grid_, shared=size_shared, stream=stream[j])


            # self.sub_compute_likelihood_1d(self.data,
            #                                self.collector_gpu_vect_frags[mode].get_ptr(),
            #                                self.gpu_sub_index,
            #                                self.gpu_sub_frag_id, self.gpu_sub_frag_len_bp, self.gpu_sub_frag_accu,
            #                                gpu_tmp_likelihood[j],
            #                                likelihood_vect,
            #                                self.gpu_param_simu,
            #                                np.int32(dim_grid), self.sub_width_mat,
            #                                self.mean_squared_frags_per_bin,
            #                                block=block_, grid=grid_, shared=size_shared, stream=stream[j])

        for j in xrange(0, self.n_modif_metropolis): # Commenting out this line should break concurrency.
            event[j]['end'].record(stream[j])
            stream[j].synchronize()
            cpu_tmp_likelihood[j] = gpu_tmp_likelihood[j].get()

        for j in xrange(0, self.n_modif_metropolis):
            score[id_x * self.n_modif_metropolis + j] = cpu_tmp_likelihood[j] + likelihood_t


    def udpate_forward_vect(self, id_fA, id_fB, id_op, max_id):
        # op 0 = pop out frag
        # op 1 = flip frag
        # op 2 = pop out insert @ left of fB orientation = 1
        # op 3 = pop out insert @ left of fB orientation = -1
        # op 4 = pop out insert @ right of fB orientation = 1
        # op 5 = pop out insert @ right of fB orientation = -1
        # print "update forward vect"
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        mode = id_op
        forward = True
        if mode < 6:
            self.pop_out_pop_in_4_mh(id_fA, id_fB, mode, max_id, forward)
        elif mode == 6 or mode == 7:
            self.split_4_mh(id_fA, max_id, forward)
        elif mode == 8:
            self.paste_4_mh(id_fA, id_fB, max_id, forward)
        elif mode > 8 and mode < 13:
            self.transloc_4_mh(id_fA, id_fB, max_id, forward)

        vect_2_copy = self.collector_gpu_vect_frags[mode]
        self.simple_copy(self.gpu_vect_frags_forward.get_ptr(), vect_2_copy.get_ptr(), self.n_frags,
                         block=block_, grid=grid_)
        forward_likelihood = self.compute_likelihood(self.gpu_vect_frags_forward, self.curr_likelihood_forward)


    def step_metropolis_hastings_s_a(self, id_fA, t, n_step, dt):
        # op 0 = pop out frag
        # op 1 = flip frag
        # op 2 = pop out insert @ left of fB orientation = 1
        # op 3 = pop out insert @ left of fB orientation = -1
        # op 4 = pop out insert @ right of fB orientation = 1
        # op 5 = pop out insert @ right of fB orientation = -1
        #####################################################################
        self.gpu_vect_frags.copy_from_gpu()
        n_contigs = len(np.unique(self.gpu_vect_frags.id_c))
        mean_len = self.gpu_vect_frags.l_cont.mean()
        max_len = self.gpu_vect_frags.l_cont.max()
        min_len = self.gpu_vect_frags.l_cont.min()
        max_id = self.modify_gl_cuda_buffer(id_fA, dt)
        V = self.jump_dictionnary[id_fA]['frags']
        V_set = self.jump_dictionnary[id_fA]['set_frags'].copy()
        id_f_left = self.gpu_vect_frags.prev[id_fA]
        id_f_right = self.gpu_vect_frags.next[id_fA]
        ori_f = self.gpu_vect_frags.ori[id_fA]
        if id_f_left != -1:
            V_set.add(id_f_left)
        if id_f_right != -1:
            V_set.add(id_f_right)
        list_neighbours = list(V_set)
        list_neighbours = np.array(list_neighbours, dtype=np.int32)
        n_neighbours = len(list_neighbours)
        F_t = self.temperature(t, n_step)
        # print "max_id = ", max_id
        #####################################################################
        forward = True
        log_score_forward = self.compute_all_score_MH(id_fA, V_set, forward)
        id_discarded_omega_fwd = self.detect_impossibility(id_fA, list_neighbours, forward)
        # print "log_score_forward = ", log_score_forward
        score_forward_T = log_score_forward / F_t
        max_score = score_forward_T.max()
        thresh_overflow = 10
        score_forward_T[score_forward_T <= max_score - thresh_overflow] = max_score - thresh_overflow
        score_forward_T = score_forward_T - score_forward_T.min()
        score_forward = np.exp(score_forward_T)
        # print "score_forward = ", score_forward
        score_forward[id_discarded_omega_fwd] = 0
        p_score_forward = score_forward / score_forward.sum()
        # print "proba forward = ", p_score_forward
        len_vect_score = n_neighbours * self.n_modif_metropolis
        # print "len vect_score = ", p_score_forward.shape
        # print "expected values = ", len_vect_score
        omega_f = np.random.choice(range(0, len_vect_score), 1, p=p_score_forward)[0]
        id_f_star = omega_f / self.n_modif_metropolis
        f_star = list_neighbours[id_f_star]
        omega_star = omega_f % self.n_modif_metropolis
        self.udpate_forward_vect(id_fA, f_star, omega_star, max_id)
        proba_forward = p_score_forward[omega_f]
        log_likelihood_star = log_score_forward[omega_f]
        #####################################################################
        forward = False
        # self.reverse_modification(list_neighbours, ori_f, id_f_left, id_f_right, f_star, omega_star)
        log_score_backward = self.compute_all_score_MH(id_fA, V_set, forward)
        id_discarded_omega_bwd = self.detect_impossibility(id_fA, list_neighbours, forward)
        target_likelihood = self.likelihood_t / F_t
        score_backward_T = log_score_backward / F_t
        max_score_back = score_backward_T.max()
        if target_likelihood <= max_score_back - thresh_overflow:
            target_likelihood = max_score_back - thresh_overflow
        score_backward_T[score_backward_T <= max_score_back - thresh_overflow] = max_score_back - thresh_overflow
        target_likelihood = target_likelihood - score_backward_T.min()
        score_backward_T = score_backward_T - score_backward_T.min()
        score_backward = np.exp(score_backward_T)
        target_likelihood = np.exp(target_likelihood)
        score_backward[id_discarded_omega_bwd] = 0
        normalization_backward = score_backward.sum()
        p_score_backward = score_backward / normalization_backward

        # id_back_modification = self.reverse_modification(list_neighbours, ori_f, id_f_left, id_f_right,
        #                                                  f_star, omega_star)
        # proba_backward = p_score_backward[id_back_modification]
        proba_backward = target_likelihood / normalization_backward
        #####################################################################
        ratio = np.exp((log_likelihood_star + proba_backward - self.likelihood_t - proba_forward) / F_t)
        r = np.min([1, ratio])
        print "ratio = ", ratio
        print "proba backward = ", proba_backward
        print "proba forward = ", proba_forward
        print "likelihood g* = ", log_likelihood_star
        print "likelihood gt = ", self.likelihood_t

        if r == 1:
            print "modification accepted"
            self.validate_struct(id_fA, f_star, omega_star, max_id)
            self.likelihood_t = log_likelihood_star
        else:
            u = np.random.rand()
            if r >= u:
                print "modification accepted"
                self.validate_struct(id_fA, f_star, omega_star, max_id)
                self.likelihood_t = log_likelihood_star
            else:
                print "modification rejected"
        dist = self.dist_inter_genome(self.gpu_vect_frags)
        return self.likelihood_t, n_contigs, min_len, mean_len, max_len, F_t, dist

    def step_mtm(self, id_fA, t, n_step, dt):
        # op 0 = pop out frag
        # op 1 = flip frag
        # op 2 = pop out insert @ left of fB orientation = 1
        # op 3 = pop out insert @ left of fB orientation = -1
        # op 4 = pop out insert @ right of fB orientation = 1
        # op 5 = pop out insert @ right of fB orientation = -1
        #####################################################################
        self.gpu_vect_frags.copy_from_gpu()
        n_contigs = len(np.unique(self.gpu_vect_frags.id_c))
        mean_len = self.gpu_vect_frags.l_cont.mean()
        max_len = self.gpu_vect_frags.l_cont.max()
        min_len = self.gpu_vect_frags.l_cont.min()
        max_id = self.modify_gl_cuda_buffer(id_fA, dt)
        V = self.jump_dictionnary[id_fA]['frags']
        V_set = self.jump_dictionnary[id_fA]['set_frags'].copy()
        id_f_left = self.gpu_vect_frags.prev[id_fA]
        id_f_right = self.gpu_vect_frags.next[id_fA]
        ori_f = self.gpu_vect_frags.ori[id_fA]
        if id_f_left != -1:
            V_set.add(id_f_left)
        if id_f_right != -1:
            V_set.add(id_f_right)
        list_neighbours = list(V_set)
        list_neighbours = np.array(list_neighbours, dtype=np.int32)
        n_neighbours = len(list_neighbours)
        F_t = self.temperature(t, n_step)
        # print "max_id = ", max_id
        #####################################################################
        forward = True
        log_score_forward = self.compute_all_score_MH(id_fA, V_set, forward)
        id_discarded_omega_fwd = self.detect_impossibility(id_fA, list_neighbours, forward)
        # print "log_score_forward = ", log_score_forward
        score_forward_T = log_score_forward / F_t
        print "score log forward = ", log_score_forward

        score_forward_T[score_forward_T == 0 ] = - np.inf
        max_score = score_forward_T.max()
        thresh_overflow = 600
        score_forward_T[score_forward_T <= max_score - thresh_overflow] = -np.inf
        # score_forward_T = score_forward_T - score_forward_T.min()
        adapt_score_fwd = np.copy(score_forward_T)
        max_forward = max_score
        print "max forward = ", max_forward
        adapt_score_fwd = adapt_score_fwd - max_forward
        adapt_score_fwd = np.exp(adapt_score_fwd)
        print "adapted score forward = ", adapt_score_fwd
        # score_forward = np.exp(score_forward_T)
        score_forward = np.copy(adapt_score_fwd)
        # print "score_forward = ", score_forward
        score_forward[id_discarded_omega_fwd] = 0


        # p_score_forward = score_forward / score_forward.sum()
        p_score_forward = score_forward / score_forward.sum()
        # print "score forward T", score_forward_T
        # print "score forward exp", score_forward
        all_score_forward = np.copy(score_forward)
        print "proba forward = ", p_score_forward
        len_vect_score = n_neighbours * self.n_modif_metropolis
        # print "len vect_score = ", p_score_forward.shape
        # print "expected values = ", len_vect_score

        omega_f = np.random.choice(range(0, len_vect_score), 1, p=p_score_forward)[0]
        id_f_star = omega_f / self.n_modif_metropolis
        f_star = list_neighbours[id_f_star]
        omega_star = omega_f % self.n_modif_metropolis
        self.udpate_forward_vect(id_fA, f_star, omega_star, max_id)
        proba_forward = p_score_forward[omega_f]
        log_likelihood_star = log_score_forward[omega_f]
        #####################################################################
        forward = False
        # self.reverse_modification(list_neighbours, ori_f, id_f_left, id_f_right, f_star, omega_star)
        V_set_back = self.return_neighbours(f_star, n_neighbours)
        log_score_backward = self.compute_all_score_MH(f_star, V_set, forward)
        id_discarded_omega_bwd = self.detect_impossibility(id_fA, list_neighbours, forward)
        target_likelihood = self.likelihood_t / F_t
        score_backward_T = log_score_backward / F_t
        score_backward_T[score_backward_T == 0] = - np.inf
        max_score_back = score_backward_T.max()
        if target_likelihood <= max_score_back - thresh_overflow:
            target_likelihood = max_score_back - thresh_overflow
        score_backward_T[score_backward_T <= max_score_back - thresh_overflow] = -np.inf
        max_backward = max_score_back
        print "max backward = ", max_backward
        target_likelihood = target_likelihood - score_backward_T.min()
        # score_backward_T = score_backward_T - score_backward_T.min()
        adapt_score_bwd = np.copy(score_backward_T)
        adapt_score_bwd = adapt_score_bwd - max_backward
        adapt_score_bwd = np.exp(adapt_score_bwd)
        print "adapted score backward = ", adapt_score_bwd
        score_backward = np.exp(score_backward_T)
        target_likelihood = np.exp(target_likelihood)
        score_backward[id_discarded_omega_bwd] = 0
        normalization_backward = score_backward.sum()
        all_score_backward = np.copy(score_backward)
        p_score_backward = score_backward / normalization_backward

        # id_back_modification = self.reverse_modification(list_neighbours, ori_f, id_f_left, id_f_right,
        #                                                  f_star, omega_star)
        # proba_backward = p_score_backward[id_back_modification]
        proba_backward = target_likelihood / normalization_backward
        #####################################################################
        # ratio = np.exp((log_likelihood_star + proba_backward - self.likelihood_t - proba_forward) / F_t)
        numerator_max = all_score_forward.max()
        denominator_max = all_score_backward.max()
        print "sum fwd  =", np.sum(adapt_score_fwd)
        print "sum bwd  =", np.sum(adapt_score_bwd)
        print "max fwd - max backward= ", np.exp(max_forward - max_backward)


        ratio = np.exp(max_forward - max_backward) * np.sum(adapt_score_fwd) / np.sum(adapt_score_bwd)

        r = np.min([1, ratio])
        print "ratio = ", ratio
        print "proba backward = ", proba_backward
        print "proba forward = ", proba_forward
        print "likelihood g* = ", log_likelihood_star
        print "likelihood gt = ", self.likelihood_t

        if r == 1:
            print "modification accepted"
            self.validate_struct(id_fA, f_star, omega_star, max_id)
            self.likelihood_t = log_likelihood_star
        else:
            u = np.random.rand()
            if r >= u:
                print "modification accepted"
                self.validate_struct(id_fA, f_star, omega_star, max_id)
                self.likelihood_t = log_likelihood_star
            else:
                print "modification rejected"
        dist = self.dist_inter_genome(self.gpu_vect_frags)
        return self.likelihood_t, n_contigs, min_len, mean_len, max_len, F_t, dist


    def detect_impossibility(self, id_fA, list_neighbours, forward):

        idx_impossibility = []
        if forward:
            gpu_vect_frags = self.gpu_vect_frags
        else:
            gpu_vect_frags = self.gpu_vect_frags_forward

        is_fA_pastable = gpu_vect_frags.prev[id_fA] == -1 or gpu_vect_frags.next[id_fA] == -1
        idx = 0
        for id_fB in list_neighbours:
            is_fB_pastable = gpu_vect_frags.prev[id_fB] == -1 or gpu_vect_frags.next[id_fB] == -1
            if not(is_fB_pastable and is_fA_pastable):
                id_paste_fB = self.n_modif_metropolis * idx + 8
                idx_impossibility.append(id_paste_fB)
            is_fB_down_splitable = gpu_vect_frags.next[id_fB] == -1
            is_fB_up_splitable = gpu_vect_frags.prev[id_fB] == -1
            id_transloc_0_fB = self.n_modif_metropolis * idx + 9
            id_transloc_1_fB = self.n_modif_metropolis * idx + 10
            id_transloc_2_fB = self.n_modif_metropolis * idx + 11
            id_transloc_3_fB = self.n_modif_metropolis * idx + 12
            if not(is_fB_down_splitable):
                idx_impossibility.append(id_transloc_0_fB)
                idx_impossibility.append(id_transloc_2_fB)
            if not(is_fB_up_splitable):
                idx_impossibility.append(id_transloc_1_fB)
                idx_impossibility.append(id_transloc_3_fB)
            idx += 1
        return idx_impossibility

    def validate_struct(self, id_fA, id_f_sampled, id_op, max_id):

        mode = id_op
        forward = True
        if mode < 6:
            self.pop_out_pop_in_4_mh(id_fA, id_f_sampled, mode, max_id, forward)
        elif mode == 6 or mode == 7:
            self.split_4_mh(id_fA, max_id, forward)
        elif mode == 8:
            self.paste_4_mh(id_fA, id_f_sampled, max_id, forward)
        elif mode > 8 and mode < 13:
            self.transloc_4_mh(id_fA, id_f_sampled, max_id, forward)
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        sampled_vect_frags = self.collector_gpu_vect_frags[mode]
        self.copy_vect(self.gpu_vect_frags.get_ptr(), sampled_vect_frags.get_ptr(), self.gpu_id_contigs, self.n_frags,
                       block=block_, grid=grid_, shared=0)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        self.init_likelihood()




    def modify_param_simu(self, param_simu, id_val, val):
        new_param_simu = np.copy(param_simu)
        if id_val == 0:
            new_param_simu["d"] = np.float32(val)
        elif id_val == 1:
            new_param_simu["slope"] = np.float32(val)

        return new_param_simu

    def step_max_likelihood_4_visu(self, id_fA, delta, size_block, dt, t, n_step):


        if id_fA not in self.id_frags_blacklisted:
            start = cuda.Event()
            end = cuda.Event()
            tic = time.time()

            self.gpu_vect_frags.copy_from_gpu()
            id_start = np.nonzero(self.gpu_vect_frags.start_bp == 0)[0]
            if np.any(self.gpu_vect_frags.start_bp < 0):
                print "problem: negative distance !!!! @ frag: ", np.nonzero(self.gpu_vect_frags.start_bp < 0)
            # if np.any(self.gpu_vect_frags.prev[id_start] != -1):
            #     print "problem: id_prev not good !!!! @ frag: ", np.nonzero(self.gpu_vect_frags.prev[id_start] != -1)
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))

            mean_len = self.gpu_vect_frags.l_cont.mean()
            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()

            self.curr_likelihood.fill(np.float64(0))
            self.gpu_vect_frags.copy_from_gpu()
            ###################################################
            block_ = (size_block, 1, 1)
            # stride = 10
            n_block = self.init_n_values_triu_extra // (size_block) + 1
            grid_all = (int(n_block / self.stride), 1)
            start.record()
            # print "dim block = ", block_
            # print "dim grid =", grid_all
            self.evaluate_likelihood(self.data,
                                     self.gpu_vect_frags.get_ptr(),
                                     self.gpu_collector_id_repeats,
                                     self.gpu_frag_dispatcher,
                                     self.gpu_sub_frag_id,
                                     self.gpu_rep_sub_frags_id,
                                     self.gpu_sub_frag_len_bp,
                                     self.gpu_sub_frag_accu,
                                     self.curr_likelihood,
                                     self.gpu_param_simu,
                                     self.init_n_values_triu,
                                     self.init_n_values_triu_extra,
                                     self.n_frags,
                                     self.init_n_sub_frags,
                                     self.mean_squared_frags_per_bin,
                                     block=block_, grid=grid_all, shared=0)
            end.record()
            end.synchronize()
            secs = start.time_till(end) * 1e-3
            # print "CUDA clock execution timing( likelihood full): ", secs
            likelihood_t = ga.sum(self.curr_likelihood, dtype=np.float64).get()
            # print "all likelihood computing time = ", time.time() - tic
            # print "current likelihood = ", likelihood_t
            self.likelihood_t = likelihood_t
            ###################################################
            tic_info = time.time()

            len_contig_A = self.gpu_vect_frags.l_cont[id_fA]

            contig_A = self.gpu_vect_frags.id_c[id_fA]
            # print "id_C contig_A = ", contig_A
            size_block_fill = 1024
            block_fill = (size_block_fill, 1, 1)
            grid_fill = (int(self.n_new_frags // size_block_fill + 1), 1)
            max_id = np.int32(ga.max(self.gpu_id_contigs).get())
            # print "max id = ", max_id
            ###################################################
            start.record()
            self.fill_sub_index_A(self.gpu_vect_frags.get_ptr(), self.gpu_sub_index, np.int32(contig_A), self.n_new_frags,
                                  block=block_fill, grid=grid_fill)
            end.record()
            end.synchronize()
            # print "fill sub index execution time = ", time.time() - tic_info
            ###################################################
            # print id_fA
            id_neighbours = self.old_return_neighbours(id_fA, delta)
            n_neighbours = len(id_neighbours)
            self.score = np.zeros((n_neighbours * self.n_tmp_struct,), dtype=np.float64)
            # print "pre score zeros = ", self.score
            ## TEST!!!!
            id_neighbours.sort() # maybe to reactivate!
            # id_neighbours.reverse() # 3d display in new_test_model
            ##########
            time_before_l = time.time()
            # print "time just before likelihood = ", time_before_l - tic
            # print "l cont fA = ", len_contig_A
            # print "n neighbors = ", n_neighbours
            for id_x in xrange(0, n_neighbours): # place where we want to spread the workload accross the network!
                self.gl_window.remote_update()
                id_fB = id_neighbours[id_x]
                # print "id_fB = ", id_fB
                self.stream_likelihood(id_fA, contig_A, len_contig_A, id_fB, id_x, likelihood_t, max_id)

            # print self.score
            # print "where = ", np.nonzero(self.score == -np.inf)
            # print "is nan = ", np.any(np.isnan(self.score))
            # print "min score = ", self.score.min()
            t_numpy = time.time()
            # print "score original = ", self.score
            ########################
            scores_2_remove = []
            scores_2_remove.extend(range(self.n_tmp_struct, len(self.score), self.n_tmp_struct)) # remove extra pop
            # scores_2_remove.extend(range(1, len(self.score), self.n_tmp_struct)) # remove all flip
            scores_2_remove.extend(range(self.n_tmp_struct + 1, len(self.score), self.n_tmp_struct)) # remove extra flip

            # for id_modif in xrange(12, self.n_tmp_struct):
            #     scores_2_remove.extend(range(self.n_tmp_struct + id_modif, len(self.score), self.n_tmp_struct)) # remove extra local flip
            # print "scores 2 remove = ", scores_2_remove
            ########################
            # print "score = ", self.score
            id_max = self.score.argmax()
            or_score = np.copy(self.score)
            filtered_score = self.score - self.score.min()
            ########################
            filtered_score[scores_2_remove] = 0
            # print "filtered score ( before) = ", filtered_score
            ########################
            max_score = filtered_score.max()
            thresh_overflow = 30
            # filtered_score[filtered_score < max_score - thresh] = 0
            filtered_score = filtered_score - (max_score - thresh_overflow)
            filtered_score[filtered_score < 0] = 0
            # print "filtered score (after)= ", filtered_score
            # id_ok_4_sampling = np.ix_(filtered_score >= max_score - thresh)
            id_ok_4_sampling = np.ix_(filtered_score > 0)
            # print "id ok for sampling = ", id_neighbours[id_ok_4_sampling[0] / self.n_tmp_struct]
            self.sub_score = filtered_score[id_ok_4_sampling]
            # print "sub score (before)= ", sub_score
            ############# DEBUGGGG ###########################
            # sub_score = np.exp(sub_score) ## debuggg !!!!
            ############# DEBUGGGG ###########################
            # print "sub score = ", sub_score
            ############# DEBUGGGG ###########################
            F_t = self.temperature(t, n_step)
            self.sub_score = self.sub_score / self.sub_score.sum()
            self.sub_score[self.sub_score > 0] = np.power(self.sub_score[self.sub_score > 0], 1./F_t)
            ############# DEBUGGGG ###########################
            self.sub_score = self.sub_score / self.sub_score.sum()
            tic_sampling = time.time()
            ### DEBUGGG #######
            if len(id_ok_4_sampling[0]) == 1 or len(id_ok_4_sampling[0]) == 0:
                sample_out = id_max
            else:
                sample_out = np.random.choice(id_ok_4_sampling[0], 1, p=self.sub_score)[0]
            ### DEBUGGG #######
            # sample_out = id_max
            ### DEBUGGG #######
            id_f_sampled = id_neighbours[sample_out / self.n_tmp_struct]
            op_sampled = sample_out % self.n_tmp_struct
            # print "id frag sampled = ", id_f_sampled
            # print "operation sampled = ", self.modification_str[op_sampled]
            # print 'id operation =', op_sampled

            self.test_copy_struct(id_fA, id_f_sampled, op_sampled, max_id)

            # print or_score
            # print sample_out
            tac = time.time()
            # print "numpy bottle time execution = ", time.time() - t_numpy
            # print "copy struct execution time = ", tac - tic_copy_struct
            # print "execution time (all)= ", tac - tic
            o = or_score[sample_out]
            self.o = o
        else:
            print "blacklist frag"
            o = self.o
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.modify_gl_cuda_buffer(id_fA, dt)

            n_contigs = len(np.unique(self.gpu_vect_frags.id_c))
            mean_len = self.gpu_vect_frags.l_cont.mean()

            mean_len_bp = self.gpu_vect_frags.l_cont_bp.mean() # test

            max_len = self.gpu_vect_frags.l_cont.max()
            min_len = self.gpu_vect_frags.l_cont.min()
            op_sampled = -1
            id_f_sampled = id_fA
            F_t = self.temperature(t, n_step)
        dist = self.dist_inter_genome(self.gpu_vect_frags)
        self.likelihood_t = o
        return o, n_contigs, min_len, mean_len, max_len_bp, op_sampled, id_f_sampled, dist, F_t
