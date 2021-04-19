import numpy as np
from typing import Union
from collections import OrderedDict, defaultdict
from slurmer.parse.metrics import Metrics


class Group:
    """
    A group of slurm jobs that share parameters (except e.g. seed)
    metrics will be computed over groups
    """

    all_param_keys = OrderedDict()
    all_result_keys = OrderedDict()

    def __init__(self, name: str, ids: list, **kwargs):
        self.name = name
        self.ids = [int(i) for i in ids]
        self.params = kwargs
        self.data = defaultdict(dict)
        self.results = OrderedDict()
        for k in kwargs.keys():
            Group.all_param_keys[k] = True

    @staticmethod
    def __filter(dct: OrderedDict, ignore_keys=()) -> OrderedDict:
        new_dct = dct.copy()
        for key in ignore_keys:
            new_dct.pop(key, None)
        # print('>>> FILTER:', 'old=', list(dct.keys()), 'new=', list(new_dct.keys()), 'ign=', ignore_keys)
        return new_dct

    @staticmethod
    def sorted_param_keys(**filter_kwargs):
        """ all known parameter keys of all groups """
        return sorted([k for k in Group.__filter(Group.all_param_keys, **filter_kwargs).keys()])

    def merge(self, other):
        """ merge another group into this one, keep this name """
        self.ids.extend(other.ids)
        self.params.update(other.params)
        self.data.update(other.data)
        self.results.update(other.results)

    def update_all_data(self, data: {str: dict}):
        """ updates the data of all group members that are in the data dict """
        for id_ in self.ids:
            if id_ in data:
                self.data[id_].update(data.get(id_))

    def update_data(self, id_: int, data: dict):
        """ updates the data of group member with slurm id """
        self.data[id_].update(data)

    def update_results(self, metrics: [Metrics]):
        for m in metrics:
            values, missing = self._values(key=m.get_key(), last_k=m.last_k)
            try:
                for result in m.from_values(values):
                    self.results[result.name] = max([result, self.results.get(result.name)])
                    Group.all_result_keys[result.name] = True
            except KeyError:
                raise KeyError('Missing key "%s" in: %s, but the metric requires it' % (m.get_key(), missing))

    def _values(self, key: str, last_k=-1) -> (Union[np.array, None], list):
        """
        all values, different group members on axis 0, time series on axis 1, (can be None)
        and a list of slurm ids where the values are missing
        """
        values = []
        missing = []
        for id_, data in self.data.items():
            if key not in data:
                missing.append(id_)
                continue
            v = np.array([v[2] for v in data.get(key)])  # tensorboard has (step, time, value) triplets
            if isinstance(last_k, int) and (last_k > 0):
                v = v[-last_k:]
            values.append(v)
        assert all([len(v) == len(values[0]) for v in values]), "different value-array lengths for key=%s" % key
        if len(values) > 0:
            return np.stack(values, axis=0), missing
        return None, missing

    def __header_dict(self, separator: str, **filter_kwargs) -> dict:
        # param_keys = Group.sorted_param_keys(**filter_kwargs)
        param_keys = list(self.__filter(self.all_param_keys, **filter_kwargs).keys())
        value_keys = list(self.__filter(self.all_result_keys, **filter_kwargs).keys())
        return {
            'n': 'name',
            'ids': 'slurm_ids',
            'params': separator.join(param_keys),
            'values': separator.join(value_keys),
        }

    def __table_dict(self, separator: str, **filter_kwargs) -> dict:
        # param_keys = Group.sorted_param_keys(**filter_kwargs)
        param_keys = list(self.__filter(self.all_param_keys, **filter_kwargs).keys())
        value_keys = list(self.__filter(self.all_result_keys, **filter_kwargs).keys())
        return {
            'n': self.name,
            'ids': str(self.ids),
            'params': separator.join([str(self.params.get(k, '')) for k in param_keys]),
            'values': separator.join([self.results.get(k).str for k in value_keys]),
        }

    def get_csv_str_header(self, **filter_kwargs) -> str:
        """ table csv header, e.g. for libre office calc """
        return '{n};{ids};;{params};;{values};'.format(**self.__header_dict(';', **filter_kwargs))

    def get_csv_str(self, **filter_kwargs) -> str:
        """ table csv row, e.g. for libre office calc, printing params and the metric values """
        return '{n};{ids};;{params};;{values};'.format(**self.__table_dict(';', **filter_kwargs))

    def get_latex_str_header(self, **filter_kwargs) -> str:
        """ table header for latex """
        return '{n} & {params} & {values} \\\\'.format(**self.__header_dict(' & ', **filter_kwargs)).replace('_', '\_')

    def get_latex_str(self, **filter_kwargs) -> str:
        """ table row for latex, printing params and the metric values """
        return '{n} & {params} & {values} \\\\'.format(**self.__table_dict(' & ', **filter_kwargs)).replace('_', '\_')


class GroupSeparator(Group):
    """
    simple hack to just insert a midrule into latex tables, and empty rows into csv data
    will probably break everything if added first to a GroupManager, so don't do that
    """
    _id = -1

    def __init__(self, **kwargs):
        self._id += 1
        super().__init__('separator %d' % self._id, [], **kwargs)

    def update_results(self, metrics):
        pass

    def get_csv_str(self, **filter_kwargs) -> str:
        """ table row for libre office calc, printing params and the metric values """
        return ''

    def get_latex_str(self, **filter_kwargs) -> str:
        """ table row for latex, printing params and the metric values """
        return '\\midrule'
