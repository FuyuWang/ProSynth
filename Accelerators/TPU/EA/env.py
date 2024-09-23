import numpy as np
import torch
import yaml
import os, sys
import copy
import random
from timeloop_env import TimeloopEnv
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import shutil
from functools import cmp_to_key, partial
from collections import defaultdict, OrderedDict
from utils import timing, is_pareto
import math
import re
import glob
import pickle
from datetime import datetime
import pandas as pd


class Environment(object):
    def __init__(self, in_config_dir='./in_config', arch_file='arch', fitness_obj=['latency'], report_dir='./report',
                 use_pool=True, use_IO=True, log_level=0, debug=False,
                 save_chkpt=False, use_sparse=True, density=None, explore_bypass=False, emulate_random=False,
                 num_pops=4):
        self.debug = bool(debug)
        self.fitness_obj = fitness_obj
        self.dim_note = ['H', 'M', 'K', 'N']
        self.len_dimension = len(self.dim_note)
        self.timeloop_configfile_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        # self.timeloop_configfile_path = f'./out_config'
        self.report_dir = report_dir
        self.use_sparse = use_sparse
        self.explore_bypass = explore_bypass
        self.density = self.get_default_density() if density is None else density
        self.timeloop_env = TimeloopEnv(config_path=self.timeloop_configfile_path, in_config_dir=in_config_dir,
                                        arch_file=arch_file, debug=self.debug,
                                        use_sparse=self.use_sparse, density=self.density)
        self.num_buf_levels = self.timeloop_env.get_num_buffer_levels()
        # print(f'Number of buffer levels: {self.num_buf_levels}')
        self.buffer_size_list = self.timeloop_env.get_buffer_size_list()
        self.buf_spmap_cstr = self.timeloop_env.get_buffer_spmap_cstr()
        self.buffers_with_spmap = list(self.timeloop_env.get_buffers_with_spmap())
        self.dimension, self.dimension_prime, self.prime2idx = self.timeloop_env.get_dimension_primes()
        self.num_primes = len(self.prime2idx.keys())
        # print(self.buf_spmap_cstr, self.buffers_with_spmap, self.buffer_size_list, self.prime2idx)
        self.use_pool = bool(use_pool)
        self.use_IO = bool(use_IO)
        self.log_level = log_level
        self.idealperf = {}

        self.set_dimension()
        self.num_pops = num_pops

        self.save_chkpt = save_chkpt
        self.fitness_record = []
        self.all_fitness_record = []
        self.sol_record = []
        self.all_sol_record = []
        self.emulate_random = emulate_random

    def get_default_density(self):
        density = {'Weights': 1,
                   'Inputs': 1,
                   'Outputs': 1}
        return density

    def set_dimension(self):
        self.idealperf['edp'], self.idealperf['latency'], self.idealperf['energy'] = self.timeloop_env.get_ideal_perf(self.dimension)
        self.idealperf['utilization'] = 1

        self.idealperf['utilization'] = 1
        self.fitness_record = []
        self.all_fitness_record = []
        self.sol_record = []
        self.all_sol_record = []

    def get_reward(self, final_trg_seq):
        if self.use_pool:
            pool = ProcessPoolExecutor(self.batch_size)
            self.timeloop_env.create_pool_env(num_pools=self.batch_size, dimension=self.dimension,
                                              sol=final_trg_seq[0, :, :], use_IO=self.use_IO)
        else:
            pool = None
            self.timeloop_env.create_pool_env(num_pools=1, dimension=self.dimension, sol=final_trg_seq[0, :, :],
                                              use_IO=self.use_IO)

        reward = self.evaluate(final_trg_seq[:, :, :], pool)

        return reward

    def mutate_thread(self, indv, alpha=1.0, beta=0.5):
        indv = self.mutate_order(indv, alpha=alpha, beta=beta)
        indv = self.mutate_tile(indv, alpha=alpha, beta=beta)
        indv = self.mutate_parallel(indv, alpha=alpha, beta=beta)
        return indv

    def thread_fun(self, args, do_mutate=True, fitness_obj=None):
        indv, pool_idx = args
        if do_mutate:
            indv = self.mutate_thread(indv)
        fit = self.timeloop_env.run_timeloop(self.dimension, indv.reshape(self.num_buf_levels*self.len_dimension, -1),
                                             pool_idx=pool_idx, use_IO=self.use_IO,
                                             fitness_obj=fitness_obj if fitness_obj is not None else self.fitness_obj)
        if do_mutate:
            return indv, fit
        else:
            return fit

    def evaluate(self, pops, pool):
        fitness = np.ones((self.num_pops, len(self.fitness_obj))) * np.NINF
        new_pops = np.empty(pops.shape)
        if not pool:
            for i, indv in enumerate(pops):
                ret = self.thread_fun((indv, 0))
                indv, fit = ret
                new_pops[i] = indv
                fitness[i] = fit
        else:
            while(1):
                try:
                    rets = list(pool.map(self.thread_fun, zip(pops, np.arange(len(pops)))))
                    for i, ret in enumerate(rets):
                        indv, fit = ret
                        new_pops[i] = indv
                        fitness[i] = fit
                    break
                except Exception as e:
                    if self.log_level>2:
                        print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(self.num_pops)
        return new_pops, fitness[:, 0]

    def select_parents(self, pops, fitness, num_parents):
        idx = np.argsort(fitness)[::-1]
        new_pops = [pops[i] for i in idx][:self.num_pops]
        new_fitness = fitness[idx][:self.num_pops]
        parents = copy.deepcopy(new_pops[:num_parents])
        return np.array(new_pops), np.array(new_fitness), np.array(parents)

    def crossover(self, parents, alpha):
        offspring_size = self.num_pops - parents.shape[0]
        offspring = np.empty((offspring_size, parents.shape[1], parents.shape[2], parents.shape[3]), dtype=np.int32)
        # for k in range(offspring_size):
        #     dad, mom = parents[random.randint(0, parents.shape[0] - 1)], parents[random.randint(0, parents.shape[0] - 1)]
        #     dad = copy.deepcopy(dad)
        #     mom = copy.deepcopy(mom)
        #     for level in range(0, self.num_buf_levels):
        #         if random.random() > 0.5:
        #             offspring[k, level, 0:self.len_dimension, 0] = dad[level, 0:self.len_dimension, 0]
        #         else:
        #             offspring[k, level, 0:self.len_dimension, 0] = mom[level, 0:self.len_dimension, 0]
        #     for d in range(self.len_dimension):
        #         if random.random() > 0.5:
        #             offspring[k, :, d, 1:] = dad[:, d, 1:]
        #             # for level in range(0, self.num_buf_levels):
        #             #     offspring[k, level, d, 0] = dad[level, d, 0]
        #         else:
        #             offspring[k, :, d, 1:] = mom[:, d, 1:]
        #             # for level in range(0, self.num_buf_levels):
        #             #     offspring[k, level, d, 0] = mom[level, d, 0]

        for k in range(offspring_size):
            parent = parents[random.randint(0, parents.shape[0] - 1)]
            offspring[k] = parent
            for pick_dim in range(self.len_dimension):
                pick_level1, pick_level2 = np.random.choice(np.arange(1, self.num_buf_levels-1), 2, replace=False)
                if random.random() > 0.5:
                    offspring[k, pick_level1, pick_dim, 0:self.num_primes+1] = parent[pick_level2, pick_dim, 0:self.num_primes+1]
                    offspring[k, pick_level2, pick_dim, 0:self.num_primes+1] = parent[pick_level1, pick_dim, 0:self.num_primes+1]

                    offspring[k, pick_level1, pick_dim, self.num_primes + 2] = min(offspring[k, pick_level1, pick_dim, 1],
                                                                                   offspring[k, pick_level1, pick_dim, self.num_primes + 2])
                    offspring[k, pick_level2, pick_dim, self.num_primes + 2] = min(offspring[k, pick_level1, pick_dim, 1],
                                                                                   offspring[k, pick_level2, pick_dim, self.num_primes + 2])

        pops = np.empty((self.num_pops,  parents.shape[1], parents.shape[2], parents.shape[3]), dtype=np.int32)
        pops[0:parents.shape[0], :] = parents
        pops[parents.shape[0]:, :] = offspring
        return pops

    def mutate_order(self, indv, alpha=1.0, beta=0.5):
        indv_mutated = copy.deepcopy(indv)
        if random.random() < alpha:
            if random.random() < beta:
                idxs = random.sample(set(np.arange(0, self.len_dimension)), 2)
                for level in range(self.num_buf_levels):
                    indv_mutated[level, idxs[0]] = indv[level, idxs[1]]
                    indv_mutated[level, idxs[1]] = indv[level, idxs[0]]
            else:
                rand_perm = np.random.permutation(self.len_dimension)
                for level in range(self.num_buf_levels):
                    for dim in range(self.len_dimension):
                        indv_mutated[level, dim] = indv[level, rand_perm[dim]]
        return indv_mutated

    def mutate_tile(self, indv, alpha=1.0, beta=0.5, num_mu_loc=3):
        if random.random() < alpha:
            if random.random() < beta:
                for _ in range(num_mu_loc):
                    pick_dim = np.random.choice(np.arange(0, self.len_dimension))
                    pick_level1, pick_level2 = np.random.choice(np.arange(1, self.num_buf_levels), 2, replace=False)
                    # pick_prime = np.random.choice(list(self.prime2idx.values()))
                    if indv[pick_level1, pick_dim, 1] == 0 or\
                            indv[pick_level1, pick_dim, 1] - 1 < indv[pick_level1, pick_dim, self.num_primes + 2]:
                        continue
                    else:
                        indv[pick_level1, pick_dim, 1] -= 1
                        indv[pick_level2, pick_dim, 1] += 1
            else:
                for pick_dim in range(self.len_dimension):
                    pick_level1, pick_level2 = np.random.choice(np.arange(1, self.num_buf_levels), 2, replace=False)
                    # pick_prime = np.random.choice(list(self.prime2idx.values()))
                    if indv[pick_level1, pick_dim, 1] == 0 or \
                            indv[pick_level1, pick_dim, 1] - 1 < indv[
                        pick_level1, pick_dim, self.num_primes + 2]:
                        continue
                    else:
                        indv[pick_level1, pick_dim, 1] -= 1
                        indv[pick_level2, pick_dim, 1] += 1
        return indv

    def mutate_parallel(self, indv, alpha=1.0, beta=0.5, num_mu_loc=3):
        if random.random() < alpha:
            if random.random() < beta:
                for _ in range(num_mu_loc):
                    pick_level = int(np.random.choice(self.buffers_with_spmap)[1:])
                    pick_dim1, pick_dim2 = np.random.choice(np.arange(0, self.len_dimension), 2, replace=False)
                    if indv[pick_level, pick_dim1, self.num_primes + 2] == 0 or \
                            indv[pick_level, pick_dim2, self.num_primes + 2] + 1 > indv[pick_level, pick_dim2, 1]:
                        continue
                    else:
                        indv[pick_level, pick_dim1, self.num_primes + 2] -= 1
                        indv[pick_level, pick_dim2, self.num_primes + 2] += 1
            else:
                for _pick_level in self.buffers_with_spmap:
                    pick_level = int(_pick_level[1:])
                    pick_dim1, pick_dim2 = np.random.choice(np.arange(0, self.len_dimension), 2, replace=False)
                    if indv[pick_level, pick_dim1, self.num_primes + 2] == 0 or \
                            indv[pick_level, pick_dim2, self.num_primes + 2] + 1 > indv[pick_level, pick_dim2, 1]:
                        continue
                    else:
                        indv[pick_level, pick_dim1, self.num_primes + 2] -= 1
                        indv[pick_level, pick_dim2, self.num_primes + 2] += 1
        return indv

    def run(self, init_pops, init_fitness, num_gens=10, parents_ratio=0.1):
        num_parents = int(self.num_pops * parents_ratio)
        pops = init_pops
        fitness = init_fitness
        if self.use_pool:
            pool = ProcessPoolExecutor(self.num_pops)
            self.timeloop_env.create_pool_env(num_pools=self.num_pops, dimension=self.dimension, sol=pops[0],
                                              use_IO=self.use_IO)
        else:
            pool = None
            self.timeloop_env.create_pool_env(num_pools=1, dimension=self.dimension, sol=pops[0], use_IO=self.use_IO)
        for g in range(num_gens):
            pops, fitness, parents = self.select_parents(pops, fitness, num_parents)
            if g == 0:
                alpha = 1
            else:
                alpha = 0.5
            pops = self.crossover(parents, alpha=alpha)
            pops, fitness = self.evaluate(pops, pool)

            best_idx = np.argmax(fitness)
            best_sol = pops[best_idx]
            print(f'[Gen{g}] fitness: {fitness[best_idx]}', fitness)
            self.record_chkpt(pops, fitness, best_idx, g, num_gens, self.num_pops)
            self.create_timeloop_report(best_sol, dir_path=self.report_dir)
            # print(f'[Gen{g}] fitness: {fitness[best_idx]} Sol: {self.get_genome(best_sol)}')
        # print(f'Achieved Fitness: {fitness[best_idx]}')

    def create_timeloop_report(self, sol, dir_path):
        fitness = self.thread_fun((sol, 0), do_mutate=False)
        stats = self.thread_fun((sol, 0), do_mutate=False, fitness_obj='all')
        os.makedirs(dir_path, exist_ok=True)
        columns = ['EDP (uJ cycles)', 'Cycles', 'Energy (pJ)', 'Utilization', 'pJ/Algorithm-Compute',
                   'pJ/Actual-Compute', 'Area (mm2)'][:len(stats)]
        if self.use_IO is False:
            self.timeloop_env.dump_timeloop_config_files(self.dimension, sol, dir_path)
        else:
            os.system(f'cp -d -r {os.path.join(self.timeloop_configfile_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path, 'RL-Timeloop.txt'), 'w') as fd:
            # print(fitness)
            value = [f'{v:.5e}' for v in fitness]
            fd.write(f'Achieved Fitness: {value}\n')
            fd.write(f'Statistics\n')
            fd.write(f'{columns}\n')
            fd.write(f'{stats}')
        stats = np.array(stats).reshape(1, -1)
        df = pd.DataFrame(stats, columns=columns)
        df.to_csv(os.path.join(dir_path, 'Gamma-Timeloop.csv'))

    def record_chkpt(self, pops, fitness, best_idx, gen, num_gens, num_pops):
        if self.save_chkpt:
            self.all_fitness_record.append(copy.deepcopy(fitness))
            self.all_sol_record.append(copy.deepcopy(pops))
            self.fitness_record.append(copy.deepcopy(fitness[best_idx]))
            self.sol_record.append(copy.deepcopy(pops[best_idx]))
            cur_gen = gen + 1
            if cur_gen == num_gens or cur_gen % 50 == 0:
                with open(os.path.join(self.report_dir, 'gamma_chkpt.plt'), 'wb') as fd:
                    chkpt = {
                        'fitness_record': self.fitness_record,
                        'all_fitness_record': self.all_fitness_record,
                        'all_sol_record': self.all_sol_record,
                        'sol_record': self.sol_record,
                        'best_fitness': self.fitness_record[-1],
                        'num_gens': num_gens,
                        'num_pops': num_pops,
                        'sampled_points': num_gens * num_pops}
                    pickle.dump(chkpt, fd)

    def clean_timeloop_output_files(self):
        shutil.rmtree(self.timeloop_configfile_path)
        out_prefix = "./timeloop-model."
        output_file_names = []
        output_file_names.append("tmp-accelergy.yaml")
        output_file_names.append(out_prefix + "accelergy.log")
        output_file_names.extend(glob.glob("*accelergy.log"))
        output_file_names.extend(glob.glob("*tmp-accelergy.yaml"))
        output_file_names.append(out_prefix + ".log")
        output_file_names.append(out_prefix + "ART.yaml")
        output_file_names.append(out_prefix + "ART_summary.yaml")
        output_file_names.append(out_prefix + "ERT.yaml")
        output_file_names.append(out_prefix + "ERT_summary.yaml")
        output_file_names.append(out_prefix + "flattened_architecture.yaml")
        output_file_names.append(out_prefix + "map+stats.xml")
        output_file_names.append(out_prefix + "map.txt")
        output_file_names.append(out_prefix + "stats.txt")
        for f in output_file_names:
            if os.path.exists(f):
                os.remove(f)









