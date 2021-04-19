from enum import Enum


class RunType(Enum):
    """
    type of the job, generally you want either FULL (actually use the desired time) or TEST (use the test queue),
    the others limit the time and are for debugging
    """

    FULL = 'full'    # full run
    TEST = 'test'    # <=15min test run
    H1 = 'h1'        # 1h test run
    H4 = 'h4'        # 4h test run
    H8 = 'h8'        # 8h test run
    H16 = 'h16'      # 16h test run
    H24 = 'h24'      # 24h test run

    @staticmethod
    def efficiency(num_gpus=1) -> float:
        """ get some time multiplier [0, 1] how much less time we need (to 0), when having multiple GPUs """
        # normalized epoch time by num gpus in DDP training, experimental results
        epoch_time = [60, 60, 35, 21, 17]
        num_gpus = min([num_gpus, len(epoch_time)-1])
        t1, tn = epoch_time[0], epoch_time[num_gpus]
        return tn / t1

    def is_test(self) -> bool:
        return self == RunType.TEST

    def scale_by_gpus(self, total_minutes: int, num_gpus: int, safety_minutes: int = 0) -> int:
        if self.is_test():
            return total_minutes
        # store safety time, for data copying
        safety_minutes = min([safety_minutes, total_minutes])
        total_minutes -= safety_minutes
        # shorten time depending on num gpus, round up to nearest 5min
        total_minutes = safety_minutes + int(total_minutes * self.efficiency(num_gpus))
        r = total_minutes % 5
        if r > 0:
            total_minutes += (5-r)
        return total_minutes

    def get_time_partition(self, test_minutes=10, days=0, hours=0, minutes=0, queue: str = None,
                           num_gpus=1, safety_minutes=0) -> (str, str):
        """
        get which partition to place the jobs on, as well as a string for the time
        you can manually override the queue to place in, otherwise pick the shortest possible one
        reduces time depending on num gpus according to training times from some personal experiments
        """
        # test / debugging
        if self == RunType.TEST:
            assert test_minutes <= 15
            return 'test', '%d:00' % test_minutes
        if self == RunType.H1:
            return 'day', '1:0:00'
        if self == RunType.H4:
            return 'day', '4:0:00'
        if self == RunType.H8:
            return 'day', '8:0:00'
        if self == RunType.H16:
            return 'day', '16:0:00'
        if self == RunType.H24:
            return 'day', '24:0:00'

        # calculate desired time and queue
        total_minutes = days * 24 * 60 + hours * 60 + minutes
        if num_gpus != 1:
            total_minutes = self.scale_by_gpus(total_minutes, num_gpus, safety_minutes=safety_minutes)
        hours, minutes = divmod(total_minutes, 60)
        days, hours = divmod(hours, 24)
        if queue is None:
            queue = 'test'
            if total_minutes > 15:
                queue = 'day'
            if total_minutes > 24*60:
                queue = 'week'
            if total_minutes > 7*24*60:
                queue = 'month'
        return queue, '%d-%d:%d:00' % (days, hours, minutes)


if __name__ == '__main__':
    a = RunType.FULL
    kwargs = dict(num_gpus=2, safety_minutes=10)

    print(a.get_time_partition(test_minutes=5, hours=15, **kwargs))
    print(a.get_time_partition(test_minutes=5, hours=15, queue='month', **kwargs))

    print()
    print(a.get_time_partition(test_minutes=5, hours=23, minutes=60, **kwargs))
    print(a.get_time_partition(test_minutes=5, hours=23, minutes=61, **kwargs))
    print(a.get_time_partition(test_minutes=5, days=3, hours=96, minutes=0, **kwargs))
    print(a.get_time_partition(test_minutes=5, days=3, hours=96, minutes=1, **kwargs))

    print('-'*10)
    for i_ in range(6):
        print(i_, a.efficiency(i_))
