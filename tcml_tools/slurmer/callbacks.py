import subprocess
import threading
from enum import Enum
from typing import Union
import time


class JobState(Enum):
    # find all states here: https://slurm.schedmd.com/sacct.html
    # these are ordered
    PENDING = 0
    RUNNING = 10
    COMPLETED = 20
    FAILED = 30
    CANCELLED = 40
    JOB_NOT_FOUND = 50

    def is_terminated(self) -> bool:
        """ whether the state is in any terminated state """
        return self in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.JOB_NOT_FOUND]

    def is_before(self, state: 'JobState') -> bool:
        """ whether the given state is in any state before the own one """
        if state.is_terminated():
            return False
        return state.value < self.value

    @classmethod
    def parse(cls, state: str) -> 'JobState':
        if state in ['PENDING']:
            return JobState.PENDING
        if state in ['COMPLETED']:
            return JobState.COMPLETED
        if state in ['RUNNING']:
            return JobState.RUNNING
        if state in ['BOOT_FAIL', 'NODE_FAIL', 'DEADLINE', 'FAILED', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT']:
            return JobState.FAILED
        if state in ['CANCELLED']:
            return JobState.CANCELLED
        raise NotImplementedError('unknown state: "%s"' % state)

    @classmethod
    def get_job_state(cls, job_id: str) -> 'JobState':
        sp = subprocess.Popen("exec sacct --format State -j %s" % job_id, shell=True, stdout=subprocess.PIPE)
        out = sp.stdout.read()
        sp.kill()
        sp.communicate()
        if len(out) == 0:
            return JobState.JOB_NOT_FOUND

        lines = str(out).split("\\n")
        if len(lines) <= 3:
            return JobState.JOB_NOT_FOUND

        lines = [line.strip() for line in lines]
        lines = lines[2:-1]  # remove table header/line in the beginning, "'" at the end
        states = [cls.parse(line) for line in lines]
        if all([s == states[0] for s in states]):
            return states[0]
        raise NotImplementedError("states differ for job_id=%s: %s" % (job_id, states))


class AbortableSleep:
    """
    A class that enables sleeping with interrupts
    see https://stackoverflow.com/questions/28478291/abortable-sleep-in-python
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._aborted = False

    def __call__(self, secs):
        with self._condition:
            self._aborted = False
            self._condition.wait(timeout=secs)
            return not self._aborted

    def abort(self):
        with self._condition:
            self._condition.notify()
            self._aborted = True


class AbstractCallback(threading.Thread):
    def __init__(self, slurm_id: Union[str, int], daemon=False, seconds=10.0):
        super().__init__()
        self.slurm_id = str(slurm_id)
        self.seconds = seconds
        self.keep_running = True
        self.daemon = daemon
        self._abortable_sleep = AbortableSleep()
        self._condition = threading.Condition()

    def wakeup(self):
        self._abortable_sleep.abort()

    def run(self):
        self.run_fun()
        while self.keep_running:
            self._abortable_sleep(self.seconds)
            self.run_fun()

    def stop(self):
        self.keep_running = False
        self._abortable_sleep.abort()

    def run_fun(self):
        with self._condition:
            self._run_fun()
            self._condition.notify()

    def _run_fun(self):
        raise NotImplementedError()


class PrintJobStateCallback(AbstractCallback):
    """ example callback for a job state """

    def _run_fun(self):
        state = JobState.get_job_state(self.slurm_id)
        print("%.2f" % time.time(), state)
        if state.is_terminated():
            print("job ended, stopping the callback")
            self.stop()


class ReachedJobStateCallback(AbstractCallback):
    """ whet the job reaches a specific state """

    def __init__(self, *args, target_state: JobState, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_state = target_state

    def _run_fun(self):
        state = JobState.get_job_state(self.slurm_id)
        if state == self.target_state:
            self.on_reach_target_state()
            self.stop()
        elif state.is_terminated() or self.target_state.is_before(state):
            self.on_reach_other_state(state)
            self.stop()

    def on_reach_target_state(self):
        print("reached target state %s, stopping" % self.target_state)

    def on_reach_other_state(self, state: JobState):
        print("reached other state %s, stopping" % state)


if __name__ == '__main__':
    PrintJobStateCallback(slurm_id="415106", seconds=1.2).start()
    # PrintJobStateCallback(slurm_id="414476", seconds=1).start()
