import os
import regex as re
import multiprocessing
from typing import Tuple, List
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator


int_pattern = re.compile('/\d+/')


class TbParser:

    @classmethod
    def list_paths(cls, dir_: str, offset=0) -> dict:
        """
        finds all tensorboard files, returns dict of {slurm_id: file path}
        requires each job to be in a subfolder named after the slurm_id
        if preceding folders are named after numbers, set the offset accordingly
        """
        file_paths = defaultdict(list)
        for sub_dir, _, file_names in os.walk(dir_):
            for file_name in file_names:
                if '.out' in file_name:
                    try:
                        ints = int_pattern.findall(sub_dir, overlapped=True)
                        id_ = int(ints[offset][1:-1])
                        file_paths[id_].append('%s/%s' % (sub_dir, file_name))
                    except:
                        pass
        return file_paths

    @classmethod
    def parse_ids(cls, dir_: str, ids: [str], **path_kwargs) -> {str: dict}:
        if len(ids) == 0:
            return {}

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        paths = cls.list_paths(dir_, **path_kwargs)

        # find+read tb data for all slurm jobs
        failed, tasks = [], []
        for id_ in ids:
            p = paths.get(id_)
            if p is None:
                failed.append(id_)
            tasks.append((id_, p))
        assert len(failed) == 0, "Can not find files to parse for slurm ids: %s" % failed
        event_tuples = pool.map(cls.read_events_files, tasks)

        # merge into dict
        events = {}
        for slurm_id, ev in event_tuples:
            events[slurm_id] = ev
        return events

    @classmethod
    def read_events_files(cls, id_filenames: Tuple[str, List[str]]) -> (str, dict):
        """
        read tensorboard files, returns all events
        adapted from https://github.com/mrahtz/tbplot/blob/master/tbplot
        """
        events = {}
        slurm_id, event_filenames = id_filenames
        for fn in event_filenames:
            try:
                ea = event_accumulator.EventAccumulator(fn)
                ea.Reload()
                for tag in ea.Tags()['scalars']:
                    events[tag] = []
                    for scalar in ea.Scalars(tag):
                        events[tag].append((scalar.wall_time, scalar.step, scalar.value))
            except Exception as e:
                print(f"While reading '{fn}':", e)
        return slurm_id, events
