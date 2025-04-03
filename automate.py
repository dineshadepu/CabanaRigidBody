#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import h5py

from itertools import cycle, product
import json
from automan.api import Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
# from pysph.solver.utils import load, get_files
from automan.api import Automator, Simulation, filter_by_name, mdict, dprod, opts2path, filter_cases
from automan.automation import (CommandTask)
from automan.api import Problem

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'sans-serif', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


# n_core = 6
n_core = 16
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params

def get_files(directory):
    # =====================================
    # start: get the files and sort
    # =====================================
    files = [filename for filename in os.listdir(directory) if filename.startswith("particles") and filename.endswith("h5") ]
    files.sort()
    files_num = []
    for f in files:
        f_last = f[10:]
        files_num.append(int(f_last[:-3]))
    files_num.sort()

    sorted_files = []
    for num in files_num:
        sorted_files.append("particles_" + str(num) + ".h5")
    files = sorted_files
    return files


class Problem01FreelyTranslatingRigidBody2D(Problem):
    """
    Function approximation
    """
    def get_name(self):
        return 'problem_01_freely_translating_rigid_body_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/problem_01_freely_translating_rigid_body_2d.py $output_dir '

        length = 2. * np.pi
        # Base case info
        self.case_info = {
            'case_1': (dict(
                length=length,
                spacing=0.3,
                hdx=1.
                ), 'dx=0.1'),
            'case_2': (dict(
                length=length,
                spacing=0.2,
                hdx=1.
                ), 'dx=0.05'),
            'case_3': (dict(
                length=length,
                spacing=0.1,
                hdx=1.
                ), 'dx=0.025'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def plot_sin_approximation(self, fname):
        for case in self.cases:
            data = np.load(case.input_path('results.npz'))

            x = data['x']
            sin_appr = data['sin_appr']
            sin_analytical = data["sin_analytical"]

            label = opts2path(case.params, keys=['spacing'])

            plt.plot(x, sin_appr, label=label.replace('_', ' = '))

        plt.plot(x, sin_analytical, label="Analytical")

        # sort the fem data before plotting

        plt.xlabel('x')
        plt.ylabel('Sin')
        plt.legend()
        plt.savefig(self.output_path(fname))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source, tail = os.path.split(self.input_path(name))

            directories = os.listdir(source)

            for directory in directories:
                try:
                    file_names = os.listdir(os.path.join(source, directory))
                    for file_name in file_names:
                        if file_name.endswith((".jpg", ".pdf", ".png")):
                            target_dir = "manuscript/figures/" + source[8:] + "/" + directory
                            try:
                                os.makedirs(target_dir)
                            except FileExistsError:
                                pass
                            shutil.copy(os.path.join(source, directory, file_name), target_dir)
                except NotADirectoryError:
                    pass

    def run(self):
        self.make_output_dir()
        self.plot_sin_approximation(fname='x_vs_sin.pdf')
        self.move_figures()


class Problem02FreelyTranslatingRigidBody3D(Problem):
    """
    Function approximation
    """
    def get_name(self):
        return 'problem_02_freely_translating_rigid_body_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/problem_02_freely_translating_rigid_body_3d.py $output_dir '

        body_length=1.
        body_height=1.
        body_depth=1.
        body_spacing=0.1
        # Base case info
        self.case_info = {
            'with_rotation_matrix': (dict(
                body_length=body_length,
                body_height=body_height,
                body_depth=body_depth,
                body_spacing=body_spacing,
                use_quaternion=0.
                ), 'Rotation Matrix'),
            'with_quaternion': (dict(
                body_length=body_length,
                body_height=body_height,
                body_depth=body_depth,
                body_spacing=body_spacing,
                use_quaternion=1.
                ), 'Quaternion'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def plot_comparision_of_rotation_matrix(self, fname):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            R_0_simu = data[name]['R_0_simu']
            time_simu = data[name]['time_simu']
            # plt.plot(time_simu, R_0_simu, label=self.cases[1])
            plt.plot(time_simu, R_0_simu, label=self.case_info[name][1])

        # sort the fem data before plotting

        plt.xlabel('Time')
        plt.ylabel('Rotation matrix index 0')
        plt.legend()
        plt.savefig(self.output_path(fname))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source, tail = os.path.split(self.input_path(name))

            directories = os.listdir(source)

            for directory in directories:
                try:
                    file_names = os.listdir(os.path.join(source, directory))
                    for file_name in file_names:
                        if file_name.endswith((".jpg", ".pdf", ".png")):
                            target_dir = "manuscript/figures/" + source[8:] + "/" + directory
                            try:
                                os.makedirs(target_dir)
                            except FileExistsError:
                                pass
                            shutil.copy(os.path.join(source, directory, file_name), target_dir)
                except NotADirectoryError:
                    pass

    def run(self):
        self.make_output_dir()
        self.plot_comparision_of_rotation_matrix(fname='time_vs_rot_mat_00.pdf')
        self.move_figures()


if __name__ == '__main__':
    PROBLEMS = [
        Problem01FreelyTranslatingRigidBody2D,
        Problem02FreelyTranslatingRigidBody3D
    ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
