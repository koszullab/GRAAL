__author__ = 'hervemn'
# coding: utf-8

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pycuda.driver as cuda
from pycuda import gpuarray as ga
import pycuda.gl as cudagl
import glutil
from vector import Vec
import numpy as np
import matplotlib.pyplot as plt
import sys
from OpenGL.arrays import vbo
from simulation_loader import simulation



class window(object):

    def __init__(self, pyramid, name, level, n_iterations, is_simu, scrambled, perform_em, sample_param, output_folder,
                 fasta_file, candidates_blacklist, n_neighbours, allow_repeats, id_selected_gpu,
                 main_thread):
        #mouse handling for transforming scene
        id_exp = 0
        self.main_thread = main_thread
        self.n_neighbours = n_neighbours
        self.id_selected_gpu = id_selected_gpu
        self.use_rippe = True
        self.mouse_down = False
        self.size_points = 6
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -2.])
        self.scrambled = scrambled
        self.sample_param = sample_param
        self.allow_repeats = allow_repeats
        self.width = 800
        self.height = 600
        self.n_iterations_em = n_iterations
        self.n_iterations_mcmc = n_iterations
        self.dt = np.float32(0.01)
        self.white = 1
        self.collect_4_graph3d = dict()
        self.output_folder = output_folder
        self.fasta_file = fasta_file
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        if perform_em:
            name_window = "Expectation Maximization : " + name
        else:
            name_window = "MCMC MTM : " + name
        name_window = "GRAAL: structure visualization"

        self.win = glutCreateWindow(name_window)

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)

        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        #setup OpenGL scene

        self.glinit()
        #setup CUDA
        self.cuda_gl_init()
        #set up initial conditions
        self.use_rippe = True

        self.simulation = simulation(pyramid, name, level, n_iterations, is_simu, self, self.output_folder,
                                     self.fasta_file, candidates_blacklist, self.allow_repeats)
        self.texid= self.simulation.texid
        self.init_n_frags = self.simulation.init_n_frags
        print "n init frags = ", self.init_n_frags
        self.pbo_im_buffer = self.simulation.pbo_im_buffer
        self.pos_vbo = self.simulation.pos_vbo
        self.col_vbo = self.simulation.col_vbo
        self.n_frags = self.simulation.sampler.n_new_frags

        print "start sampling... "
        self.str_likelihood = "likelihood = " + str(0)
        self.str_n_contigs = "n contigs = " + str(0)
        self.str_curr_id_frag = "current id frag = " + str(0)
        self.str_curr_cycle = "current cycle = "+ str(0)
        self.str_curr_temp = "current temperature = "+ str(0)
        self.str_curr_dist = "current dist = "+ str(0)

        self.collect_likelihood = []
        self.collect_n_contigs = []
        self.collect_mean_len = []
        self.collect_op_sampled = []
        self.collect_id_fA_sampled = []
        self.collect_id_fB_sampled = []
        self.collect_full_likelihood = []
        self.collect_dist_from_init_genome = []
        self.collect_fact = []
        self.collect_slope = []
        self.collect_d = []
        self.collect_d_nuc = []
        self.collect_d_max = []
        self.collect_likelihood_nuisance = []
        self.collect_success = []
        self.collect_all = []

        self.file_mean_len = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_mean_len.pdf')
        self.file_n_contigs = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_n_contigs.pdf')

        self.file_fact = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_fact.pdf')
        self.file_slope = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_slope.pdf')
        self.file_d_nuc = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d_nuc.pdf')
        self.file_d = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d.pdf')
        self.file_d_max = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d_max.pdf')

        self.file_dist_init_genome = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_dist_init_genome.pdf')

        self.txt_file_mean_len = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_mean_len.txt')
        self.txt_file_n_contigs = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_n_contigs.txt')
        self.txt_file_dist_init_genome = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_dist_init_genome.txt')
        self.txt_file_likelihood = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_likelihood.txt')

        self.txt_file_fact = os.path.join(self.simulation.output_folder, str(id_exp) +  'list_fact.txt')
        self.txt_file_slope = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_slope.txt')
        self.txt_file_d_nuc = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d_nuc.txt')
        self.txt_file_d = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d.txt')
        self.txt_file_d_max = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d_max.txt')
        self.txt_file_success = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_success.txt')
        self.txt_file_list_mutations = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_mutations.txt')
        self.file_all_data = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_all.txt')

    def replay_simu(self, folder_res,):


        file_simu = os.path.join(folder_res, 'list_mutations.txt')
        file_likelihood = os.path.join(folder_res, 'list_likelihood.txt')
        file_n_contigs = os.path.join(folder_res, 'list_n_contigs.txt')
        file_distances = os.path.join(folder_res, 'list_dist_init_genome.txt')

        h = open(file_simu, 'r')
        all_lines = h.readlines()
        list_id_fA = []
        list_id_fB = []
        list_op_sampled = []
        for i in xrange(1, len(all_lines)):
            line = all_lines[i]
            data = line.split('\t')
            id_fA = int(data[0])
            id_fB = int(data[1])
            id_op = int(data[2])
            list_id_fA.append(id_fA)
            list_id_fB.append(id_fB)
            list_op_sampled.append(id_op)
        h.close()

        h_likeli = open(file_likelihood, 'r')
        all_lines_likeli = h_likeli.readlines()
        list_likelihood = []
        for i in xrange(0, len(all_lines_likeli)):
            line = all_lines_likeli[i]
            likeli = np.float32(line)
            list_likelihood.append(likeli)
        h_likeli.close()

        h_n_contigs = open(file_n_contigs, 'r')
        list_n_contigs = []
        all_lines_contigs = h_n_contigs.readlines()
        for i in xrange(0, len(all_lines_contigs)):
            line = all_lines_contigs[i]
            n_contigs = np.float32(line)
            list_n_contigs.append(n_contigs)
        h_n_contigs.close()

        h_distances = open(file_distances, 'r')
        list_distances = []
        all_lines_distances = h_distances.readlines()
        for i in xrange(0, len(all_lines_distances)):
            line = all_lines_distances[i]
            distance = np.float32(line)
            list_distances.append(distance)
        h_distances.close()

        for i in xrange(0, 1000):
            self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
            self.remote_update()
            self.str_n_contigs = "n contigs = " + str(self.simulation.sampler.gpu_vect_frags.id_c.max())

        self.simulation.sampler.explode_genome(self.dt)
        # print list_likelihood
        for i in xrange(0, len(list_id_fA)):
            self.simulation.sampler.apply_replay_simu(list_id_fA[i], list_id_fB[i], list_op_sampled[i], self.dt)
            self.str_curr_cycle = "cycle = " + str(int(i))
            self.str_likelihood = "likelihood = " + str(list_likelihood[i])
            self.str_n_contigs = "n contigs = " + str(list_n_contigs[i])
            self.str_curr_id_frag = "current frag = "+ str(list_id_fA[i])
            self.str_curr_dist = "current dist = "+ str(list_distances[i])
            self.str_curr_temp = "current temperature = "+str(0)

        self.simulation.export_new_fasta()


    def start_EM(self,):
        print "start expectation maximization ... "
        delta = np.ones((self.n_iterations_em,),dtype=np.int32) * self.n_neighbours
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        ready = 0

        if self.scrambled:
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.scrambled_input_matrix)
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        self.iter = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        for j in xrange(0, self.n_iterations_em):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                self.curr_frag = i
                if bool(glutMainLoopEvent):
                    glutMainLoopEvent()
                else:
                    glutCheckLoop()

                o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist, temp = self.simulation.sampler.step_max_likelihood(i,
                                                                                                 delta[j],
                                                                                                 512, self.dt,
                                                                                                 np.float32(j), n_iter)

                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_temp = "current temperature = "+str(temp)
                self.str_curr_d = "current d = " + str(d)

                self.collect_full_likelihood.append(self.simulation.sampler. likelihood_t)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)
                self.collect_dist_from_init_genome.append(dist)
                self.iter += 1
                # sampling nuisance parameters
                if self.sample_param :
                    fact, d, d_max, d_nuc, slope, likeli, success, y_eval = self.simulation.sampler.step_nuisance_parameters(self.dt, np.float32(j),
                                                                                                    n_iter)
                    self.y_eval = y_eval
                else:
                    success = 1
                    curr_param = np.copy(self.simulation.sampler.param_simu)
                    kuhn, lm, c1, slope, d, d_max, fact, d_nuc = curr_param[0]
                    likeli = o
                mean_len_kb = mean_len / 1000.
                self.main_thread.update_gui(o, n_contigs, mean_len_kb, dist, slope, d_max)

                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)

            o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)

        self.simulation.export_new_fasta()
        self.save_behaviour_to_txt()

    def init_4_sub_sampling(self, fact, id_exp):

        self.simulation.output_folder = os.path.join(self.simulation.folder_sub_sampling, str(fact))
        if not os.path.exists(self.simulation.output_folder):
            os.mkdir(self.simulation.output_folder)




        self.file_mean_len = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_mean_len.pdf')
        self.file_n_contigs = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_n_contigs.pdf')

        self.file_fact = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_fact.pdf')
        self.file_slope = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_slope.pdf')
        self.file_d_nuc = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d_nuc.pdf')
        self.file_d = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d.pdf')
        self.file_d_max = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_d_max.pdf')

        self.file_dist_init_genome = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_dist_init_genome.pdf')

        self.txt_file_mean_len = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_mean_len.txt')
        self.txt_file_n_contigs = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_n_contigs.txt')
        self.txt_file_dist_init_genome = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_dist_init_genome.txt')
        self.txt_file_likelihood = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_likelihood.txt')

        self.txt_file_fact = os.path.join(self.simulation.output_folder, str(id_exp) +  'list_fact.txt')
        self.txt_file_slope = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_slope.txt')
        self.txt_file_d_nuc = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d_nuc.txt')
        self.txt_file_d = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d.txt')
        self.txt_file_d_max = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_d_max.txt')
        self.txt_file_success = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_success.txt')
        self.txt_file_list_mutations = os.path.join(self.simulation.output_folder, str(id_exp) + 'list_mutations.txt')
        self.file_all_data = os.path.join(self.simulation.output_folder, str(id_exp) + 'behaviour_all.txt')

        self.simulation.sampler.update_texture_4_sub(fact)

    def save_behaviour_to_txt(self):
        list_file = [self.txt_file_mean_len, self.txt_file_n_contigs, self.txt_file_dist_init_genome, self.txt_file_likelihood,
                     self.txt_file_fact, self.txt_file_slope, self.txt_file_d_max, self.txt_file_d_nuc,
                     self.txt_file_success]
        list_data = [self.collect_mean_len, self.collect_n_contigs, self.collect_dist_from_init_genome, self.collect_likelihood,
                     self.collect_fact, self.collect_slope, self.collect_d_max, self.collect_d_nuc,
                     self.collect_success]
        for d in range(0, len(list_file)):
            thefile = list_file[d]
            h = open(thefile, 'w')
            data = list_data[d]
            for item in data:
                h.write("%s\n" % item)
            h.close()
        f_mutations = open(self.txt_file_list_mutations, 'w')
        f_mutations.write("%s\t%s\t%s\n"%("id_fA", "id_fB", "id_mutation"))
        for i in xrange(0, len(self.collect_id_fA_sampled)):
            id_fA = self.collect_id_fA_sampled[i]
            id_fB = self.collect_id_fB_sampled[i]
            id_mut = self.collect_op_sampled[i]
            f_mutations.write("%s\t%s\t%s\n"%(id_fA, id_fB, id_mut))
        f_mutations.close()

    def start_MTM(self,):
        print "set jumping distribution..."
        delta = 5
        self.simulation.sampler.set_jumping_distributions_parameters(delta)
        self.simulation.sampler.init_likelihood()
        print "start sampling launched ... "
        print self.simulation.n_iterations
        delta = range(5, 5 + self.simulation.n_iterations * 2, 2)
        print delta
        # if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
        # o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        n_iter = np.float32(self.simulation.n_iterations)
        list_frags = np.arange(0, self.n_frags, dtype=np.int32)
        for j in xrange(0, self.n_iterations_mcmc):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            for i in list_frags:
                # print "id_frag =", i
                if bool(glutMainLoopEvent):
                    glutMainLoopEvent()
                else:
                    glutCheckLoop()
                o, n_contigs, min_len, mean_len, max_len, temp, dist = self.simulation.sampler.step_mtm(i, np.float32(j), n_iter, self.dt)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_temp = "current temperature = "+ str(temp)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_dist_from_init_genome.append(dist)

        # self.simulation.export_new_fasta()
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_mcmc)

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
                                       "n_contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
                                       "mean length contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
                                       "distance from init genome")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
                                       "slope")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
                                       "scale factor")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
                           "val trans")

        self.save_behaviour_to_txt()


    def remote_update(self):
        if bool(glutMainLoopEvent):
            glutMainLoopEvent()
        else:
            glutCheckLoop()

    def setup_simu(self, id_f_ins):
        self.simulation.sampler.insert_repeats(id_f_ins)
        self.simulation.sampler.simulate_rippe_contacts()
        plt.imshow(self.simulation.sampler.hic_matrix, interpolation='nearest', vmin=0, vmax=100)
        plt.show()

    def test_model(self, id_fi, delta):
        id_fi = np.int32(id_fi)
        id_neighbors = np.copy(self.simulation.sampler.old_return_neighbours(id_fi, delta))
        id_neighbors.sort()
        np.sort(id_neighbors)
        print "physic model = ", self.simulation.sampler.param_simu
        j = 0
        n_iter = self.n_iterations_em
        self.simulation.sampler.step_max_likelihood_4_visu(id_fi, delta, 512, self.dt, np.float32(j), n_iter)
        nscore = np.copy(self.simulation.sampler.score)
        plt.figure()
        plt.plot(id_neighbors, nscore[range(0, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(1, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(2, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(3, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(4, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(5, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(6, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(7, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(12, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)

        plt.legend([self.simulation.sampler.modification_str[i] for i in range(0, 8)])
        plt.ylabel('log likelihood')
        plt.xlabel('fragments id')
        # plt.show()
        plt.figure()
        for i in range(0, self.simulation.sampler.n_tmp_struct):
            print i
            plt.plot(id_neighbors, nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.legend(self.simulation.sampler.modification_str)
        plt.show()

    def discrete_spiral(self, n_points):
        # (di, dj) is a vector - direction in which we move right now
        di = 1
        dj = 0
        # length of current segment
        segment_length = 1

        # current position (i, j) and how much of current segment we passed
        i = 0
        j = 0
        segment_passed = 0

        output_x = [0]
        output_y = [0]
        for k in range(0, n_points - 1):
            # make a step, add 'direction' vector (di, dj) to current position (i, j)
            i += di
            j += dj
            output_x.append(i)
            output_y.append(j)

            segment_passed += 1
            if (segment_passed == segment_length):
                # done with current segment
                segment_passed = 0

                # 'rotate' directions
                buffer = di
                di = -dj
                dj = buffer

                # increase segment length if necessary
                if (dj == 0):
                    segment_length += 1
        return output_x, output_y



    def new_test_model(self, id_fi, vmin, vmax):

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.mlab import griddata
        delta = self.simulation.sampler.n_frags - 1
        id_fi = np.int32(id_fi)
        id_neighbors = np.copy(self.simulation.sampler.old_return_neighbours(id_fi, delta))
        # id_neighbors.revert()

        id_neighbors_sorted = np.sort(id_neighbors)
        arg_index = np.argsort(id_neighbors)
        list_arg_index = list(arg_index)

        # np.sort(id_neighbors)
        print "physic model = ", self.simulation.sampler.param_simu
        j = 0

        n_iter = self.n_iterations_em
        self.simulation.sampler.step_max_likelihood_4_visu(id_fi, delta, 512, self.dt, np.float32(j), n_iter)
        ori_nscore = np.copy(self.simulation.sampler.score)


        n_activ_frags = len(ori_nscore) / self.simulation.sampler.n_tmp_struct
        n_tmp_struct = self.simulation.sampler.n_tmp_struct
        nscore = np.zeros_like(ori_nscore, dtype=np.float64)

        id = 0
        max_id = n_activ_frags - 1
        for i in list_arg_index:
            id_start = id * n_tmp_struct
            id_end = (id + 1) * n_tmp_struct
            loc_index = range((max_id - i) * n_tmp_struct, (max_id - i + 1) * n_tmp_struct)
            # print loc_index
            nscore[loc_index] = ori_nscore[id_start: id_end]
            id += 1
        nscore = np.array(nscore, dtype=np.float64)
        # print nscore
        m = np.min(nscore)
        print "val min score = ", m
        print "val max score = ", np.max(nscore)
        id_start = 0
        new_index = self.simulation.sampler.return_neighbours(id_fi, delta)
        new_index.reverse()
        init_ordered_score = nscore[range(7, len(nscore), self.simulation.sampler.n_tmp_struct)] - m
        out_x, out_y = self.discrete_spiral(len(init_ordered_score))
        # big_x, big_y = self.discrete_spiral(self.simulation.sampler.n_tmp_struct - id_start)

        big_x = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        big_y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]




        x = []
        y = []
        z = []
        dt = 0.1
        out_x = np.array(out_x, dtype=np.float32)
        out_y = np.array(out_y, dtype=np.float32)
        dx_span = 2 * out_x.max()
        dy_span = 2 * out_y.max()

        # ord_mut = range(id_start, self.simulation.sampler.n_tmp_struct)
        ord_mut = [6, 7, 2, 3, 4, 5, 8, 9, 10, 11]
        ord_mut = ord_mut.reverse()

        # ord_mut = [6]
        # for i in range(0, len(ord_mut)):
        #     x.extend(list(out_x + big_x[i] * dx_span))
        #     y.extend(list(out_y + big_x[i] * dy_span))
        #     s = nscore[range(ord_mut[i], len(nscore), self.simulation.sampler.n_tmp_struct)] - m
        #     z.extend(list(s))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # print "len out x", len(out_x)
        # ax.plot_trisurf(x, y, z, cmap=plt.cm.jet)
        # plt.show()
        print "ord mut =", ord_mut
        print "len nscore= ", nscore.shape
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        t = 0
        if vmin == '':
            Tvmin = (nscore.max()-m)*0.001
            Tvmax = (nscore.max()-m)
        else:
            m = vmin
            Tvmin = 0
            Tvmax = vmax - vmin

        for i in ord_mut:

            # x.extend(list(out_x + big_x[i - id_start] * dx_span))
            # y.extend(list(out_y + big_y[i - id_start] * dy_span))
            x.extend(list(out_x + big_x[t] * dx_span))
            y.extend(list(out_y + big_y[t] * dy_span))

            s = nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)] - m
            # print "mutation group = ", i
            # print "local min = ", s.min()
            z.extend(list(s + m))
            loc_x = out_x + big_x[t] * dx_span
            loc_y = out_y + big_y[t] * dy_span
            # xi = np.linspace(loc_x.min(), loc_x.max(), 800)
            # yi = np.linspace(loc_y.min(), loc_y.max(), 800)
            # zi = griddata(loc_x, loc_y, s, xi, yi)

            # tri = ax.plot_trisurf(loc_x, loc_y, s, cmap=plt.cm.gist_rainbow_r, vmin=Tvmin, vmax=Tvmax)

            # xv, yv = np.meshgrid(xi, yi)
            # cset = ax.contourf(xv, yv, zi, zdir='z', offset = -50, cmap=plt.cm.gist_rainbow_r, vmin=0, vmax=nscore.max() - m)

            t += 1

        # plt.colorbar(tri)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        xi = np.linspace(x.min(), x.max(), 400)
        yi = np.linspace(y.min(), y.max(), 400)
        zi = griddata(x, y, z, xi, yi)
        xv, yv = np.meshgrid(xi, yi)
        # ax.plot_wireframe(xv, yv, zi)
        # cset = ax.contourf(xv, yv, zi, zdir='y', offset = 50, cmap=plt.cm.gist_rainbow_r)


        # ax.set_zlabel('Log likelihood')
        # ax.set_axis_off()
        # ax.set_zlim(bottom=Tvmin, top=Tvmax, auto=False)
        # plt.show()

        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, c=z, cmap=plt.cm.gist_rainbow_r, vmin=vmin, vmax=vmax)
        # print "vmin = ", (nscore.max()-m)*0.5
        # print "vmax = ", (nscore.max()-m)
        # plt.show()
        #
        # im_out = np.ones((max(x) - min(x) + 1, max(y) - min(y) + 1), dtype=np.float32) * m
        # x = np.array(x)
        # y = np.array(y)
        # im_out[list(x - x.min()), list(y - y.min())] = z
        # plt.imshow(im_out, cmap = plt.cm.gist_rainbow_r, interpolation='nearest',  vmin=(nscore.max()-m)*0.5, vmax=nscore.max()-m)
        # plt.colorbar()
        # plt.show()


        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.bar3d(x, y, np.zeros(len(x)), 1, 1, z, cmap=plt.cm.jet)
        # plt.show()
        #


        # plt.figure()
        # plt.plot(id_neighbors, nscore[range(0, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(1, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(2, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(3, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(4, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(5, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(6, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(7, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # # plt.plot(id_neighbors, nscore[range(12, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        #
        # plt.legend([self.simulation.sampler.modification_str[i] for i in range(0, 8)])
        # plt.ylabel('log likelihood')
        # plt.xlabel('fragments id')
        # # plt.show()
        # plt.figure()
        # for i in range(0, self.simulation.sampler.n_tmp_struct):
        #     print i
        #     plt.plot(id_neighbors, nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.legend(self.simulation.sampler.modification_str)
        # plt.show()
        return id_neighbors, nscore, x, y, z



    def debug_test_model(self, id_fi, delta):
        id_fi = np.int32(id_fi)
        id_neighbors = np.copy(self.simulation.sampler.return_neighbours(id_fi, delta))
        id_neighbors.sort()
        np.sort(id_neighbors)
        print "physic model = ", self.simulation.sampler.param_simu
        self.simulation.sampler.debug_step_max_likelihood(id_fi, delta, 512, self.dt)
        nscore = np.copy(self.simulation.sampler.score)
        plt.figure()
        plt.plot(id_neighbors, nscore[range(0, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(1, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(2, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(3, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(4, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(5, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(6, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.legend([self.simulation.sampler.modification_str[i] for i in range(0, 7)])
        # plt.show()
        plt.figure()
        for i in range(0, self.simulation.sampler.n_tmp_struct):
            print i
            plt.plot(id_neighbors, nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.legend(self.simulation.sampler.modification_str)
        plt.show()

    def cuda_gl_init(self,):
        cuda.init()
        if bool(glutMainLoopEvent):
            # print "----SELECT COMPUTING DEVICE (NVIDIA GPU)----"
            # num_gpu = cuda.Device.count()
            # for i in range(0,num_gpu):
            #     tmp_dev = cuda.Device(i)
            #     print "device_id = ",i,tmp_dev.name()
            # id_gpu = raw_input("Select GPU: ")
            # id_gpu = int(id_gpu)
            curr_gpu = cuda.Device(self.id_selected_gpu)
            print "you have selected ", curr_gpu.name()
            self.ctx_gl = cudagl.make_context(curr_gpu, flags=cudagl.graphics_map_flags.NONE)
        else:
            import pycuda.gl.autoinit
            curr_gpu = cudagl.autoinit.device
            self.ctx_gl = cudagl.make_context(curr_gpu, flags=cudagl.graphics_map_flags.NONE)

    def glut_print(self, x,  y,  font,  text, r,  g , b , a):

        blending = False
        if glIsEnabled(GL_BLEND) :
            blending = True

        glEnable(GL_BLEND)
        glColor3f(r, g, b)
        glRasterPos2f(x, y)
        for ch in text :
            glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )


        if not blending :
            glDisable(GL_BLEND)

    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., self.width / float(self.height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            self.simulation.release()
            sys.exit()
        elif args[0] == 'p':
            self.size_points += 1
        elif args[0] == 'm':
            self.size_points -= 1
        elif args[0] == 's':
            self.start_EM()
        elif args[0] == 'w':
            self.white *= -1
        # elif args[0] == 'e':
        #     # idn0, nscore0, x0, y0, z0 = self.new_test_model(self.curr_frag, '', '')
        #     # self.collect_4_graph3d[self.iter] = dict()
        #     # self.collect_4_graph3d[self.iter]['x0'] = x0
        #     # self.collect_4_graph3d[self.iter]['y0'] = y0
        #     # self.collect_4_graph3d[self.iter]['z0'] = z0
        #
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
        #                                    "n_contigs")
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
        #                                    "mean length contigs")
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
        #                                "distance from init genome")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
        #                                "slope")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
        #                                "scale factor")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
        #                    "val trans")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d, self.file_d,
        #                    "d")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_max, self.file_d_max,
        #                    "dist max intra")
        #
        #     # if self.sample_param:
        #     if 1 == 1:
        #         plt.figure()
        #         plt.loglog(self.bins_rippe, self.y_eval, '-b')
        #         plt.loglog(self.bins_rippe, self.simulation.sampler.mean_contacts, '-*r')
        #         plt.xlabel("genomic separation ( kb)")
        #         plt.ylabel("n contacts")
        #         plt.title("rippe curve")
        #         plt.legend(["fit", "obs"])
        #         plt.show()
        #     self.simulation.sampler.display_modif_vect(0, 0, -1, 100)
        #     self.save_behaviour_to_txt()
        #     self.simulation.export_new_fasta()
        #     o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y


    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS

    def draw(self):
        """Render the particles"""
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.white == -1:
            glClearColor(1, 1, 1, 1)
        else:
            glClearColor(0, 0, 0, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)

        #render the particles
        self.render_image()
        self.render()

        #draw the x, y and z axis as lines
        glutil.draw_axes()
        ############## enable 2d display #######################
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, self.width, self.height, 0.0, -1.0, 10.0)
        glMatrixMode(GL_MODELVIEW)

        glLoadIdentity()
        glDisable(GL_CULL_FACE)

        glClear(GL_DEPTH_BUFFER_BIT)

        self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 0.0 , 1.0 , 0.0 , 1.0)
        self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 0.0 , 1.0 , 0.0 , 1.0)
        self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 0.0 , 1.0 , 0.0 , 1.0)
        self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 0.0 , 1.0 , 0.0 , 1.0)
        self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 0.0 , 1.0 , 0.0 , 1.0)
        self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 0.0 , 1.0 , 0.0 , 1.0)

        # self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10.1 , 60.1 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 9.9 , 59.9, GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 1.0 , 1.0 , 1.0 , 1.0)


        ## Making sure we can render 3d again
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        #############
        glutSwapBuffers()


    def render(self):

        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glBegin(GL_POINTS)


        # glEnable(GL_POINT_SMOOTH)

        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST) # a reactiver


        # glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        #
        glPointSize(self.size_points)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)
        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, int(self.simulation.sampler.n_new_frags))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        # glEnd(GL_POINTS)

        glDisable(GL_BLEND)

    def render_image(self):
        blending = False
        if glIsEnabled(GL_BLEND):
            blending = True
        else:
            glEnable(GL_BLEND)

        glColor4f(1, 1, 1, 1)

        # glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, self.texid)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_im_buffer)

        #Copyng from buffer to texture
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.init_n_frags, self.init_n_frags, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind

        #glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        #

        glBegin(GL_QUADS)

        glVertex2f(-1, 0)
        glTexCoord2f(-1, 0)
        glVertex2f(0, 0)
        glTexCoord2f(0, 0)
        glVertex2f(0, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glTexCoord2f(-1, 1)

        glEnd()
        glDisable(GL_TEXTURE_2D)

        if not blending :
            glDisable(GL_BLEND)

