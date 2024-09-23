import copy
import os
import argparse

import numpy as np
import yaml
import pickle

from env import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="energy", help='1st order fitness objective')
    parser.add_argument('--fitness2', type=str, default=None, help='2nd order fitness objective')
    parser.add_argument('--fitness3', type=str, default=None, help='3rd order fitness objective')
    parser.add_argument('--num_pops', type=int, default=2,help='number of populations')
    parser.add_argument('--num_gens', type=int, default=2, help='number of generations/epochs')
    parser.add_argument('--config_path', type=str, default='../in_config',
                        help='Configuration path, should include arch.yaml, problem.yaml, (and sparse.yaml if sparsity is considered)')
    parser.add_argument('--report_dir', type=str, default='../report', help='The report directory')
    parser.add_argument('--density', type=str, default='0.5,1,1', help='The density of Input, Output, Weight Tenor')
    parser.add_argument('--save_chkpt', action='store_true', default=False, help='Create a checkpoint when finished')
    parser.add_argument('--use_sparse', action='store_true', default=False, help='Execute Map Space Exploration on sparse accelerator')
    parser.add_argument('--explore_bypass', action='store_true', default=False,
                        help='Enable it can add bypass buffer option in to the search space')

    parser.add_argument('--dnn', type=str, default=None, help='dnn model')
    parser.add_argument('--input_size', type=int, default=1, help='the size of dimension N')
    parser.add_argument('--architecture', type=str, default='arch', help='accelerator architecture')

    opt = parser.parse_args()
    fitness = [opt.fitness1]
    fitness.append(opt.fitness2) if opt.fitness2 is not None else None
    fitness.append(opt.fitness3) if opt.fitness3 is not None else None
    print(f'Fitness Objective: {fitness}')
    density = opt.density.split(',')
    density = {'Inputs': float(density[0]), 'Outputs': float(density[1]), 'Weights': float(density[2])}
    problem_dir = '../../../Benchmarks'

    architectures = ['nvdla']
    for architecture in architectures:
        llms = ['bertlarge']
        for llm in llms:
            with open(os.path.join(problem_dir, '{}_problems/layers.yaml'.format(llm)), 'r') as fd:
                layers = yaml.load(fd, Loader=yaml.SafeLoader)
            fd.close()
            problem = {'problem': {'instance': {'H': 1, 'M': 512, 'K': 1024, 'N': 1024},
                                   'shape': {'data-spaces':
                                                 [{'name': 'Weights', 'projection': [[['H']], [['K']], [['N']]]},
                                                  {'name': 'Inputs', 'projection': [[['H']], [['M']], [['K']]]},
                                                  {'name': 'Outputs', 'projection': [[['H']], [['M']], [['N']]],
                                                   'read-write': True}],
                                             'dimensions': ['H', 'M', 'K', 'N'], 'name': 'bmm'}}}
            # input_sizes = [1, 4, 16, 64, 256, 1024]
            input_sizes = [1]

            for input_size in input_sizes:
                actor_state_dict = None
                layer2chkpt = {}
                for i, layer in enumerate(layers[1:2]):
                    print(architecture, llm, input_size, i, layer)
                    report_dir = os.path.join(opt.report_dir, architecture, 'EA_fitness_{}'.format(opt.fitness1),
                                              'sampled_episodes_{}'.format(opt.num_pops),
                                              '{}_input{}'.format(llm, input_size), 'layer-{}'.format(i))
                    with open(os.path.join(problem_dir, '{}_problems/{}.yaml'.format(llm, layer)), 'r') as fd:
                        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
                        problem['problem']['instance']['H'] = layer_problem['problem']['H'] * input_size
                        problem['problem']['instance']['M'] = layer_problem['problem']['M']
                        problem['problem']['instance']['K'] = layer_problem['problem']['K']
                        problem['problem']['instance']['N'] = layer_problem['problem']['N']
                    fd.close()

                    layer_to_key = ''
                    for key in ['H', 'M', 'K', 'N']:
                        layer_to_key += str(problem['problem']['instance'][key]) + ' '

                    if layer_to_key in layer2chkpt:
                        print(layer_to_key, 'repeated')
                        chkpt = layer2chkpt[layer_to_key]
                        os.makedirs(report_dir, exist_ok=True)
                        with open(os.path.join(report_dir, 'env_chkpt.plt'), 'wb') as fd:
                            pickle.dump(chkpt, fd)
                        fd.close()
                        continue
                    else:
                        with open('../in_config/problem.yaml', 'w') as fd:
                            yaml.dump(problem, fd)
                        fd.close()

                        env = Environment(fitness_obj=fitness, report_dir=report_dir, use_pool=True, use_IO=True,
                                          debug=False, in_config_dir=opt.config_path, arch_file=architecture,
                                          density=density, save_chkpt=opt.save_chkpt,
                                          use_sparse=opt.use_sparse, explore_bypass=opt.explore_bypass,
                                          num_pops=opt.num_pops)

                        # pop = np.array([[0, 0, 0, 0],
                        #                 [1, 0, 0, 0],
                        #                 [3, 2, 0, 0],
                        #                 [2, 0, 0, 0],
                        #                 [0, 0, 0, 0],
                        #                 [1, 4, 1, 4],
                        #                 [3, 5, 0, 0],
                        #                 [2, 5, 1, 4],
                        #                 [0, 0, 0, 0],
                        #                 [1, 2, 1, 2],
                        #                 [3, 3, 0, 0],
                        #                 [2, 0, 0, 0],
                        #                 [0, 0, 0, 0],
                        #                 [1, 3, 0, 0],
                        #                 [3, 0, 0, 0],
                        #                 [2,5 ,0,0]]).reshape((4, 4, 4))

                        pop = np.array([[0, 0, 0, 0],
                                         [1, 0, 0, 0],
                                         [3, 0, 0, 0],
                                         [2, 0, 0, 0],
                                         [0, 1, 1, 1],
                                         [1, 5, 1, 5],
                                         [3, 4, 0, 0],
                                         [2, 5, 1, 2],
                                         [0, 1, 1, 1],
                                         [1, 4, 0, 0],
                                         [3, 5, 0, 0],
                                         [2, 1, 1, 1],
                                         [0, 2, 0, 0],
                                         [1, 0, 0, 0],
                                         [3, 0, 0, 0],
                                         [2, 0, 0, 0]]).reshape((4, 4, 4))

                        # pop = np.array([[0, 0, 0, 0, 0, 0],
                        #                 [1, 0, 0, 0, 0, 0],
                        #                 [3, 1, 0, 0, 0, 0],
                        #                 [2, 0, 0, 0, 0, 0],
                        #                 [0, 0, 0, 0, 0, 0],
                        #                 [1, 6, 0, 1, 6, 0],
                        #                 [3, 3, 0, 0, 0, 0],
                        #                 [2, 7, 0, 1, 2, 0],
                        #                 [0, 0, 0, 0, 0, 0],
                        #                 [1, 4, 0, 0, 0, 0],
                        #                 [3, 2, 1, 1, 2, 0],
                        #                 [2, 0, 0, 0, 0, 0],
                        #                 [0, 0, 0, 0, 0, 0],
                        #                 [1, 1, 0, 0, 0, 0],
                        #                 [3, 6, 0, 0, 0, 0],
                        #                 [2, 5, 1, 0, 0, 0]]).reshape((4, 4, 6))

                        # pop = np.array([[0, 0, 0, 0, 0, 0],
                        #                 [1, 0, 0, 0, 0, 0],
                        #                 [3, 3, 0, 0, 0, 0],
                        #                 [2, 0, 0, 0, 0, 0],
                        #                 [0, 3, 0, 0, 0, 0],
                        #                 [1, 4, 0, 1, 3, 0],
                        #                 [3, 1, 0, 1, 1, 0],
                        #                 [2, 5, 0, 1, 4, 0],
                        #                 [0, 1, 1, 0, 0, 0],
                        #                 [1, 1, 0, 1, 1, 0],
                        #                 [3, 1, 0, 0, 0, 0],
                        #                 [2, 1, 0, 1, 1, 0],
                        #                 [0, 1, 0, 0, 0, 0],
                        #                 [1, 6, 0, 0, 0, 0],
                        #                 [3, 6, 0, 0, 0, 0],
                        #                 [2, 1, 0, 0, 0, 0]]).reshape((4, 4, 6))
                        init_pops = np.zeros((opt.num_pops, pop.shape[0], pop.shape[1], pop.shape[2]), dtype=np.int32)
                        for p in range(0, opt.num_pops):
                            init_pops[p] = copy.deepcopy(pop)
                        # init_fitness = np.full(opt.num_pops, -666430211.096576)
                        init_fitness = np.full(opt.num_pops, -415792598.188032)
                        # init_fitness = np.full(opt.num_pops, -242116629784367.16)
                        # init_fitness = np.full(opt.num_pops, -18637182104953.16)
                        env.run(init_pops, init_fitness, num_gens=opt.num_gens)

                        env.clean_timeloop_output_files()
