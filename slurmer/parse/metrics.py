import numpy as np


class Result:
    """
    A result value paired with float accuracy for string representation
    This class is used by Group, no need to create instances of it elsewhere
    """

    def __init__(self, name: str, value, float_acc=2):
        self.name = name
        self.value = value
        self.fp_str = '%.' + str(float_acc) + 'f'

    @property
    def str(self) -> str:
        if isinstance(self.value, str):
            return self.value
        return self.fp_str % self.value

    def __lt__(self, other: 'Result') -> bool:
        if other is None:
            return False
        if isinstance(self.value, str):
            return True
        if isinstance(other.value, str):
            return False
        return self.value < other.value

    def __eq__(self, other: 'Result') -> bool:
        if other is None:
            return False
        if isinstance(self.value, str) and isinstance(other.value, str):
            return True
        return self.value == other.value


class Metrics:
    """
    Metrics that apply to 2D (jobs, time) numpy arrays
    """

    def __init__(self, key: str, name: str = None, not_available_value: str = None,
                 first_k=None, last_k=None, float_acc: int = 2, replace_nan: float = None,
                 avg=False, med=False, max_avg=False, min_avg=False,
                 std=False, max=False, min=False):
        """
        :param key: matching the tensorboard str, e.g. "test/accuracy"
        :param name: name of the metric, use 'key' as name if None
        :param not_available_value: value to return if it is not possible to perform a metric calculation
        :param first_k: number of last epochs to consider (good for avg/std), None to consider all (good for max/min)
        :param last_k: number of last epochs to consider (good for avg/std), None to consider all (good for max/min)
        :param float_acc: for the string representation of float numbers
        :param replace_nan: if not None: set all nan-values to this value
        :param avg: set to True if you want to calculate this metric (mean/average)
        :param med: set to True if you want to calculate this metric (median)
        :param max_avg: set to True if you want to calculate this metric (max value of each job, averaged)
        :param min_avg: set to True if you want to calculate this metric (min value of each job, averaged)
        :param std: set to True if you want to calculate this metric (standard deviation)
        :param max: set to True if you want to calculate this metric (maximum)
        :param min: set to True if you want to calculate this metric (minimum)
        """
        assert (first_k is None) or (last_k is None), "Can not use both first_k and last_k"

        self.key = key
        self.name = name if isinstance(name, str) else key
        self.not_available_value = not_available_value
        self.first_k = first_k
        self.last_k = last_k
        self.float_acc = float_acc
        self.replace_nan = replace_nan

        self.avg = avg
        self.med = med
        self.max_avg = max_avg
        self.min_avg = min_avg
        self.std = std
        self.max = max
        self.min = min

    def get_key(self) -> str:
        """ the key in the log file """
        return self.key

    def from_values(self, values: np.ndarray = None) -> [Result]:
        """
        Calculates the desired metrics from the given values

        :param values: values of different group members on axis 0, time series on axis 1
        :return: list of results
        """
        if isinstance(values, np.ndarray):
            v = values[:, :self.first_k] if self.last_k is not None else values
            v = v[:, -self.last_k:] if self.last_k is not None else v
            if self.replace_nan is not None:
                v = np.nan_to_num(v, copy=True, nan=self.replace_nan)
        elif values is None:
            assert self.not_available_value is not None,\
                "(%s | %s) Can not calc metrics (None array) and no not_available_value" % (self.key, self.name)
            v = None
        else:
            raise NotImplementedError("undefined data type %s: %s" % (type(values), repr(values)))

        results = []
        if self.avg:
            results.append(Result(
                '%s avg' % self.name,
                self.not_available_value if v is None else np.mean(v),
                float_acc=self.float_acc))
        if self.med:
            results.append(Result(
                '%s med' % self.name,
                self.not_available_value if v is None else np.median(v),
                float_acc=self.float_acc))
        if self.max_avg:
            results.append(Result(
                '%s max_avg' % self.name,
                self.not_available_value if v is None else np.mean(np.max(v, axis=1), axis=0),
                float_acc=self.float_acc))
        if self.min_avg:
            results.append(Result(
                '%s min_avg' % self.name,
                self.not_available_value if v is None else np.mean(np.min(v, axis=1), axis=0),
                float_acc=self.float_acc))
        if self.std:
            results.append(Result(
                '%s std' % self.name,
                self.not_available_value if v is None else np.std(np.mean(v, axis=1), axis=0),
                float_acc=self.float_acc))
        if self.max:
            results.append(Result(
                '%s max' % self.name,
                self.not_available_value if v is None else np.max(v),
                float_acc=self.float_acc))
        if self.min:
            results.append(Result(
                '%s min' % self.name,
                self.not_available_value if v is None else np.min(v),
                float_acc=self.float_acc))
        return results
