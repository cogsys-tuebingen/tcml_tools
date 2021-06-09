"""
parse metrics from tensorboard files alone!
- find all tensorboard (tb) files in the given directory
- define metrics over values of interest (e.g. test accuracy)
- parse tb files and calculate all metrics
- can group slurm jobs that only differ by e.g. seeds, can filter for specific job arguments (e.g. used model)
- summarize metrics in a table (calc or latex)
"""


import os
import shutil
import shelve
import itertools
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from tcml_tools.slurmer.parse import Group, GroupSeparator, TbParser, Result


class GroupManager:
    """
    Organize multiple groups, filter them, compare them against each other, get structured results
    """

    def __init__(self, storage_path: str = None):
        self.groups = []
        self.empty_result = Result(name="__gm_empty__", value="N/A", float_acc=0)
        self.stored_data = dict(events={})
        self.storage_path = storage_path
        self.open_storage()

    def open_storage(self):
        """ open a shelve storage, current storage must be empty for that """
        if isinstance(self.storage_path, str):
            assert (self.stored_data is None)\
                   or isinstance(self.stored_data, shelve.DbfilenameShelf)\
                   or (len(self.stored_data['events']) == 0)
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self.stored_data = shelve.open(self.storage_path, writeback=True)
            self.stored_data['events'] = self.stored_data.get('events', {})

    def close_storage(self, use_dict_now=False):
        """ close the current shelve storage """
        if isinstance(self.stored_data, shelve.DbfilenameShelf):
            events = self.get_all_events() if use_dict_now else None
            self.stored_data.sync()
            self.stored_data.close()
            if use_dict_now:
                self.stored_data = dict(events={})
                self.stored_data['events'] = events

    def get_all_events(self, copy=False) -> {str: dict}:
        """ get all stored events """
        if copy:
            return self.stored_data['events'].copy()
        return self.stored_data['events']

    def update_events(self, events: {str: dict}):
        """ update the stored events with further ones """
        self.stored_data['events'].update(events)
        if isinstance(self.stored_data, shelve.DbfilenameShelf):
            self.stored_data.sync()

    def copy(self):
        """
        close the storage and return a copy of this manager
        (necessary since only one manager can concurrently access the storage file)
        """
        self.close_storage(use_dict_now=True)
        return deepcopy(self)

    def merge(self, other: 'GroupManager'):
        """ merge another GroupManager into this one, each group is merged by name """
        self.update_events(other.get_all_events())
        for g1 in self.get_groups():
            for g2 in other.get_groups():
                if g1.name == g2.name:
                    g1.merge(g2)
                    break

    def merge_groups_by_param(self, name: str, as_new_groups=False):
        """
        merge all groups in this GroupManager that only differ by one parameter, removes separators

        :param name:
        :type name:
        :param as_new_groups: keep the individual groups and the merged ones, otherwise keep only the merged ones
        """
        # cluster groups, remove separators
        clusters = defaultdict(list)

        for g in self.get_groups():
            if isinstance(g, GroupSeparator):
                continue
            clusters[g.get_param_tuple(skip_keys=(name,))].append(g)
        # merge clusters
        groups = []
        for cluster in clusters.values():
            g = Group("merged %d" % len(cluster), ids=[])
            for g2 in cluster:
                g.merge(g2)
            g.params.pop(name)
            groups.append(g)
        # insert / replace
        if as_new_groups:
            self.groups.extend(groups)
        else:
            self.groups = groups
            for g in self.groups:
                g.params[name] = ""
            Group.all_param_keys.pop(name)

    def add_group(self, *args, **kwargs):
        """ create and add a Group """
        self.groups.append(Group(*args, **kwargs))

    def add_existing_groups(self, groups: list):
        self.groups.extend(groups)

    def add_separator(self):
        """ add a separator, just for printing data/tables """
        self.groups.append(GroupSeparator())

    def find_directories(self, dir_, **path_kwargs) -> {int: str}:
        """ find the corresponding sub-directories of the given jobs """
        paths = TbParser.list_paths(dir_, **path_kwargs)
        sub_dirs = {}
        for g in self.get_groups():
            for id_ in g.ids:
                path = paths.get(id_)
                if isinstance(path, str):
                    path_parts = path.split(str(id_))
                    path_parts[-1] = ''
                    path = str(id_).join(path_parts) + '/'
                    sub_dirs[id_] = path
        return sub_dirs

    def delete_on_disk(self, dir_: str, **path_kwargs):
        """ deletes all related directories on the disk, cannot be undone """
        sub_dirs = self.find_directories(dir_, **path_kwargs)
        for id_, path in sub_dirs.items():
            if isinstance(path, str):
                shutil.rmtree(path, ignore_errors=True)
                print(id_, "\tremoved\t", path)

    def copy_folders(self, source_dir: str, target_dir: str, add_group_name=True, **path_kwargs):
        """ copies all related directories on the disk to a specified dir, optionally grouped by group name """
        sub_dirs = self.find_directories(source_dir, **path_kwargs)
        for g in self.get_groups():
            target_dir_ = '%s/%s/' % (target_dir, g.name) if add_group_name else target_dir
            for id_ in g.ids:
                target_dir__ = '%s/%d/' % (target_dir_, id_)
                path = sub_dirs.get(id_)
                if isinstance(path, str):
                    shutil.copytree(path, target_dir__)
                    print(id_, '\tcopied\t', path, '\t->\t', target_dir__)

    def get_groups(self, copy=False, remove_separators=False) -> [Group]:
        """ get all groups """
        groups = [g for g in self.groups if not isinstance(g, GroupSeparator)] if remove_separators else self.groups
        if copy and not remove_separators:
            groups = groups.copy()
        return groups

    def get_group(self, name=None, job_id=None) -> Group:
        """ return a copy of an existing group, by name (str) or job_id (int) """
        assert name is not None or job_id is not None
        for g in self.get_groups():
            if name == g.name or job_id in g.ids:
                return deepcopy(g)

    def copy_group(self, name: str, new_name: str, param_changes: dict):
        """ create and add a copy of an existing group, modify the params  """
        g = self.get_group(name=name)
        g.name = new_name
        g.params.update(param_changes)
        self.add_existing_groups([g])

    def filter_groups(self, filters: dict, keep_separators=True) -> 'GroupManager':
        """
        use only subset of groups which match the filters, filters in form of {key: [values]}
        if you want to use another subset of groups later, use this on a copy of your main GroupManager
        """
        new_groups = []
        for g in self.get_groups():
            add = True
            for key, values in filters.items():
                if g.params.get(key, None) not in values:
                    add = False
                    break
            if add or (keep_separators and isinstance(g, GroupSeparator)):
                new_groups.append(g)
        self.groups = new_groups
        return self

    def update_groups(self, dir_: str, metrics: list, force=False, **path_kwargs) -> 'GroupManager':
        """
        reads the tb files for all jobs in all groups, which are found in dir_, updates the groups
        unless 'force' is set to True, used cached events info
        """
        # figure out which ids need parsing
        ids = []
        for group in self.get_groups():
            ids.extend(group.ids)

        # only parse newly added jobs
        if not force:
            events = self.get_all_events(copy=False)
            new_ids = []
            for id_ in ids:
                if len(events.get(id_, {})) == 0:
                    new_ids.append(id_)
            ids = new_ids

        # parse all events, add to storage
        events = TbParser.parse_ids(dir_, ids=ids, **path_kwargs)
        self.update_events(events)

        # update data and results of each group
        for g in self.get_groups():
            g.update_all_data(self.get_all_events())
            g.update_results(metrics)
        return self

    def get_values(self, key: str, unique=True) -> list:
        """ get (different) values for a given param/result key """
        values = []
        for group in self.get_groups():
            values.append(group.get(key, "N/A"))
        if unique:
            values = list(set(values))
        return values

    @staticmethod
    def _align_strs(strs: [str], align_symbol='&') -> [str]:
        """ align strings (table style) by column wise by the given alignment symbol """
        split_strs = [s.split(align_symbol) for s in strs]
        lengths = [0 for _ in range(len(split_strs[0]))]
        for s in split_strs:
            for i, s2 in enumerate(s):
                lengths[i] = max(lengths[i], len(s2))
        combination_str = (' %s ' % align_symbol).join(['{%d:<%d}' % (i, l) for i, l in enumerate(lengths)])
        return [s[0] if len(s) == 1 else combination_str.format(*s) for s in split_strs]

    def sorted_results(self, sort_by: str, descending: bool, no_consecutive_separators=True) -> [Group]:
        results = self.get_groups(copy=True)
        if len(sort_by) > 0:
            results = sorted(results, key=lambda g: g.results.get(sort_by, self.empty_result), reverse=descending)
        if no_consecutive_separators:
            separator_last, separator_cur = False, False
            results2 = []
            for groups in results:
                separator_last, separator_cur = separator_cur, isinstance(groups, GroupSeparator)
                if not (separator_last and separator_cur):
                    results2.append(groups)
            results = results2
        return results

    def print_csv_table(self, sort_by='', descending=True, **filter_kwargs):
        """
        prints csv text, e.g. to copy to libre office calc, optionally sorted by some metric result
        """
        print(self.get_groups()[0].get_csv_str_header(**filter_kwargs))
        for g in self.sorted_results(sort_by, descending, no_consecutive_separators=True):
            print(g.get_csv_str(**filter_kwargs))

    def print_latex_table(self, sort_by='', descending=True, **filter_kwargs):
        """
        prints text to copy to latex, optionally sorted by some metric result
        the tables are intended for the latex booktabs package
        """
        strs = [self.get_groups()[0].get_latex_str_header(**filter_kwargs)]
        groups = self.sorted_results(sort_by, descending, no_consecutive_separators=True)
        for g in groups:
            strs.append(g.get_latex_str(**filter_kwargs))
        strs = self._align_strs(strs, '&')
        print('\\begin{tabular}{l%s}' % ('c'*strs[0].count('&')))
        print('\\toprule')
        print(strs[0])
        print('\\midrule')
        for s in strs[1:]:
            print(s)
        print('\\bottomrule')
        print('\\end{tabular}')

    def get_data_frame(self) -> pd.DataFrame:
        """ get a pandas DataFrame of the parsed data """
        data = {}
        for key in itertools.chain(Group.all_param_keys, Group.all_result_keys):
            data[key] = [g.get(key) for g in self.get_groups(copy=False, remove_separators=True)]
        return pd.DataFrame(data)

    def get_plot_data(self, combine: str, on_x_axis: str, on_y_axis: str) -> dict:
        """
        get data that can be plot easily

        :param combine: key, which groups have the same "line"
        :param on_x_axis: key, which param/result to put on the x axis
        :param on_y_axis: key, which param/result to put on the y axis
        :return: dict of the following structure:
                {
                    key0: [(x00, y00), (x01, y01), ..., (x0n, y0n)],
                    key1: [(x10, y10), (x11, y11), ..., (x1m, y1m)],
                    ...
                }
        """
        clusters = defaultdict(list)

        # get desired values, grouped
        for g in self.get_groups():
            if isinstance(g, GroupSeparator):
                continue
            c = g.params.get(combine, "__missing_key__")
            clusters[c].append((g.get(on_x_axis, default=-1), g.get(on_y_axis, default=-1)))

        # sort ascending
        for k in clusters.keys():
            clusters[k] = sorted(clusters[k], key=lambda v: v[0])

        return clusters
