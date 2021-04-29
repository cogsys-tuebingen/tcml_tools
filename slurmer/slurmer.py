"""
interface to easily manage slurm jobs
- create sbatch+sh files from templates, parametrize your runs via a grid search
- queue the created files
- print a csv table of the slurm ids + parameter combinations
- print copy-paste code to easily fetch results later via GroupManager (slurmer.parse)

maybe in future:
- await jobs to finish
    - optionally delete sbatch/sh/.err/.out files
- continuously log metrics of interest, such as cpu/memory usage
    - https://stackoverflow.com/questions/24020420/find-out-the-cpu-time-and-memory-usage-of-a-slurm-job
"""


import os
import time
import subprocess
import re
import shutil
from typing import Callable
from collections import Iterable, OrderedDict


int_pattern = re.compile('\d+')


class JobItem:
    def __init__(self, job_name: str, param_names: list, param_combination: list):
        self.job_name = job_name
        self.param_names = param_names
        self.param_combination = param_combination
        self.sb_file = None
        self.sh_file = None
        self.slurm_id = -1

    def params_str(self, num=-1):
        return '_'.join([str(s) for s in self.param_combination[:num]])


class Slurmer:
    def __init__(self,
                 copy_path=os.path.dirname(__file__),
                 copy_file_sb='template.sbatch',
                 copy_file_sh='template.sh',
                 paste_path='/tmp/tcml_tools/generated/'):
        """
        slurm management via python script

        :param copy_path: path to the templates
        :param copy_file_sb: name of the sbatch template
        :param copy_file_sh: name of the shell template
        :param paste_path: where to put all the generated files
        """
        self.copy_path = os.path.expanduser(copy_path + '/')
        self.copy_file_sb = copy_file_sb
        self.copy_file_sh = copy_file_sh
        self.paste_path = os.path.expanduser(paste_path + '/')

        # currently fixed replacement strings
        self.str_replace_params_in_sh = '{params}'               # params in the sh file (grid search)
        self.str_replace_sharedvalues_in_sh = '{shared_values}'  # shared values in the sh file (same for all)
        self.str_replace_in_sb_path = '{path}'                   # sh path in the sbatch file
        self.str_replace_name = '{name}'                         # experiment name in the sbatch+sh files
        self.str_sh_lines_before = '{lines_before}'              # additional commands before the python3 statement
        self.str_sh_lines_after = '{lines_after}'                # additional commands after the python3 statement

        # set/added/changed by user
        self.str_replacements = {
            'job_name': 'name_not_set',
            'nodes': '1',
            'ntasks': '1',
            'cpus_per_task': '4',
            'mem_per_cpu': '12G',
            'gpus': '1',
            'partition': 'test',
            'time': '00:15:00',
            'out_path': './out/',
            'mail_type': 'ALL',
        }
        self.sh_params_names = []                                # list of param names in the grid search
        self.sh_param_values = []                                # this is a list to ensure name ordering
        self.sh_lines_before = []
        self.sh_lines_after = []
        self.sh_shared_values = {}

        # all jobs, not necessarily started via slurm
        self.jobs = []

    def set_str_replacements(self, replacements: dict):
        """ replaces '{x}' with 'ABCD' when given {'x': 'ABCD'} in the templates' content """
        self.str_replacements.update(replacements)

    def set_sh_shared_values(self, fixed_values: dict):
        """ adds '--key=value' or '--key' when value is None """
        self.sh_shared_values = fixed_values

    def set_param_values(self, param_values: list):
        """ given a list of params like [('name', [1, 2])] for the param 'name' there will be the two values [1, 2]"""
        self.sh_param_values = param_values

    def add_sh_line(self, text: str, before=True, after=False):
        """ add a line to the sh file, before the python3 command and/or after it """
        if before:
            self.sh_lines_before.append(text)
        if after:
            self.sh_lines_after.append(text)

    def add_param(self, name: str, values: list):
        """ extending the list of params """
        self.sh_param_values.append((name, values))

    def add_job(self, job_name: str, variables: list):
        """ this requires that params have been set! variables are taken in order of param names """
        self.jobs.append(JobItem(job_name=job_name, param_names=self.sh_params_names, param_combination=variables))

    @staticmethod
    def _safe_str(vars_) -> list:
        """ prevent e.g. file paths from messing with the resulting file names """
        safe_vars = []
        for v in vars_:
            if isinstance(v, Iterable) and not isinstance(v, str):
                safe_vars.append(Slurmer._safe_str(v))
            else:
                if isinstance(v, str) and '/' in v:
                    splits = v.split('/')
                    for s in reversed(splits):
                        p = int_pattern.findall(s)
                        if len(p) > 0:
                            v = p[-1]
                            break
                safe_vars.append(v)
        return safe_vars

    def _grid(self, *args) -> list:
        """ returns a list of: list of vars; grid search """
        if len(args) == 1:
            return [[v] for v in args[0]]

        grids = self._grid(*args[:-1])
        new_grids = []
        for v in args[-1]:
            for g in grids:
                new_grids.append(g + [v])
        return new_grids

    def _named_grid(self, name_pattern: str, *args):
        """ returns a list of: dict of: {applied namepattern : var list} """
        grid = self._grid(*args)
        return zip([name_pattern.format(*self._safe_str(r_)) for r_ in grid], grid)

    def _fully_replace(self, s: str) -> str:
        for k, v in self.str_replacements.items():
            s = s.replace('{%s}' % k, str(v))
        return s

    def create_files(self, search_grid=True, verbose=True):
        """
        Creates sbatch and sh files in the target dir
        :param search_grid: adds a gridsearch over all given parameters to the jobs
        :param verbose: whether to print each created file with params/path
        :return:
        """
        print('\nCreating files')
        # create folders
        if self.str_replacements.get('out_path', None) is not None:
            path = os.path.expanduser(self.str_replacements.get('out_path').replace("${HOME}", "~"))
            path = os.path.dirname(path)
            os.makedirs(self._fully_replace(path), exist_ok=True)
        self.copy_path = os.path.expanduser(self.copy_path)
        self.paste_path = os.path.expanduser(self.paste_path)
        os.makedirs(self.paste_path, exist_ok=True)

        # patterns
        self.sh_params_names = list([p[0] for p in self.sh_param_values])
        params_sh_str = ' '.join('--%s={%i}' % (name, i) for i, name in enumerate(self.sh_params_names))
        name_pattern = '_'.join(['{%i}' % i for i in range(len(self.sh_param_values))])

        # remove any grid search keys from the defaults
        for name in self.sh_params_names:
            if name in self.sh_shared_values:
                self.sh_shared_values.pop(name)

        # add grid search jobs
        if search_grid and len(self.sh_param_values) > 0:
            for n, v in self._named_grid(name_pattern, *[p[1] for p in self.sh_param_values]):
                self.jobs.append(JobItem(n, self.sh_params_names, v))

        # cache template contents
        with open(self.copy_path + self.copy_file_sh) as f:
            content_sh = f.read()
            content_sh = content_sh.replace(self.str_sh_lines_before, '\n'.join(self.sh_lines_before))
            content_sh = content_sh.replace(self.str_sh_lines_after, '\n'.join(self.sh_lines_after))
            sh_shared = [(k, '=', v) if v is not None else (k, '', '') for k, v in self.sh_shared_values.items()]
            sh_shared = ' '.join(['--%s%s%s' % v for v in sh_shared])
            content_sh = content_sh.replace(self.str_replace_sharedvalues_in_sh, sh_shared)
            content_sh = self._fully_replace(content_sh)
        with open(self.copy_path + self.copy_file_sb) as f:
            content_sb = f.read()
            content_sb = self._fully_replace(content_sb)

        for i, job_item in enumerate(self.jobs):
            paste_file_sh = self.paste_path + job_item.job_name + '.sh'
            paste_file_sb = self.paste_path + job_item.job_name + '.sbatch'
            set_params_sh_str = params_sh_str + ''
            for j, value in enumerate(job_item.param_combination):
                set_params_sh_str = set_params_sh_str.replace('{%d}' % j, str(value))

            # SH file
            content_sh_ = content_sh.replace(self.str_replace_params_in_sh, set_params_sh_str)
            content_sh_ = content_sh_.replace(self.str_replace_name, job_item.job_name)
            # write to new file
            with open(paste_file_sh, "w+") as f:
                f.write(content_sh_)

            # SBATCH file
            content_sb_ = content_sb.replace(self.str_replace_in_sb_path, paste_file_sh)
            content_sb_ = content_sb_.replace(self.str_replace_name, job_item.job_name)
            # write to new file
            with open(paste_file_sb, "w+") as f:
                f.write(content_sb_)

            # set permissions, add to queue
            os.system("chmod 775 " + paste_file_sh)
            os.system("chmod 775 " + paste_file_sb)
            job_item.sh_file = paste_file_sh
            job_item.sb_file = paste_file_sb

        if verbose:
            max_len = max([len(str(job_item.param_combination)) for job_item in self.jobs])
            format_str = '{0:<%d}{1}' % (max_len+4)
            print(format_str.format('variables', 'file path'))
            for job_item in self.jobs:
                print(format_str.format(str(job_item.param_combination), job_item.sb_file))
        print('Created 2*%d files in %s' % (len(self.jobs), self.paste_path))

    def queue_files(self):
        """
        Queues all created files with the slurm workload manager,
        waits for files to exist
        :return:
        """
        print('\nQueueing %d jobs now' % len(self.jobs))
        for i, job_item in enumerate(self.jobs):
            while not (os.path.exists(job_item.sb_file) and os.path.exists(job_item.sh_file)):
                time.sleep(0.1)
            out = subprocess.Popen("sbatch %s" % job_item.sb_file, shell=True, stdout=subprocess.PIPE).stdout.read()
            if len(out) == 0:
                print('Failed queueing job #%d, no output' % i)
            else:
                job_item.slurm_id = int_pattern.findall(out.decode('utf-8'))[-1]

    def delete_files(self, folder=True):
        """ careful, jobs that have not yet started will crash due to this """
        if folder:
            shutil.rmtree(self.paste_path, ignore_errors=True)
        else:
            for job_item in self.jobs:
                os.remove(job_item.sh_file)
                os.remove(job_item.sb_file)

    def print_table(self, separator=';'):
        """ Prints the jobs in a table format, which is easy to copy-paste into calc/excel """
        print('\nJobs in table form:')
        print('%s%s%s' % ('slurm_id', separator, separator.join(self.sh_params_names)))
        for job_item in self.jobs:
            print('%s%s%s' % (job_item.slurm_id, separator, separator.join([str(v) for v in job_item.param_combination])))

    def print_group_code(self, ignore=('seed', 'note'), cast_params: Callable = None):
        """
        Prints the jobs in format for slurmer.parse

        :param ignore: list/tuple of parameter keys that may differ within each group
        :param cast_params: optional Callable fun(param name, param value) that returns a tuple of (name, value)
                            to modify the printed lines
        """
        print('\nJobs in group code form:')
        group_ids = OrderedDict()
        group_params = OrderedDict()

        def f(vars_: list) -> (str, str):
            not_ignored = []
            for p, v in zip(self.sh_params_names, vars_):
                if p not in ignore:
                    not_ignored.append((p, "'%s'" % v) if type(v) is str else (p, v))
            if isinstance(cast_params, Callable):
                not_ignored = [cast_params(k_, v_) for k_, v_ in not_ignored]
            index_str = '#'.join([str(v) for _, v in not_ignored])
            params_str = ', '.join(['%s=%s' % (u, v) for u, v in not_ignored])
            return index_str, params_str

        for job_item in self.jobs:
            s1, s2 = f(job_item.param_combination)
            if group_ids.get(s1) is None:
                group_ids[s1] = []
            group_ids[s1].append(job_item.slurm_id)
            group_params[s1] = s2

        for i, k in enumerate(group_ids.keys()):
            print("gm.add_group('n{i}', {ids}, {params})".format(**{
                'i': i,
                'ids': [int(i) for i in group_ids[k]],
                'params': group_params[k],
            }))
