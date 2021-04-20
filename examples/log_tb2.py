"""
parse completed jobs

TODO:
- set the path, which directory each job saves to ("save_dir" value in log_tb1.py)
    - since I personally execute the script from my local machine, the path accounts for the sshfs mounting
    - if you do run this from the cluster as well, the path should start with /home/...
- paste the "gm.add_group(...)" output lines and remove their placeholder in 3.
- be sure that the pythonpath is set
- cd to the script and run: python3 log_tb2.py
"""


import os
from slurmer.parse import GroupManager, Metrics


if __name__ == "__main__":
    # 1. TODO figure out which directory everything is saved to.
    # it is necessary that the runs are jobs save into dirs depending on their slurm id (happens by default)
    # generally you probably want to mount the TCML cluster, e.g. via SSHFS
    path = ""
    # path = '/mnt/tcml-master01/mnt/beegfs/home/<my-username>/experiments/tcml_tools/log_tb/'  # example
    path = os.path.expanduser(path)

    # 2. define a group manager
    gm = GroupManager()
    # you could make the GM cache its info by adding a storage file path, speeding it up when re-computing known groups

    # 3. groups
    # define groups of slurm jobs, metrics will average over them
    # add key:value pairs as you want, they can be used to filter, sort, and are printed in the table later
    # TODO insert your groups from the previous python script here, remove the placeholder
    gm.add_group('n0', [1234], mu=0.1, model='resnet18')

    # 4. metrics
    # define the metrics used to parse your results
    # they are extracted from the tensorboard files that are found in the given path and referenced to a specific job
    # the keys are exactly what is published in tensorboard
    # adapt as needed for your specific use case
    metrics = [
        # the only thing we logged to tensorboard, use the last 10 values, compute avg & med & std
        # since r depends on the seed, the numbers should come out the same
        Metrics('random/r', float_acc=2, last_k=10, avg=True, med=True, std=True, max=True, min=True),

        # this metric uses a non-existent key in the data, and publishes under another name,
        # but only for values that do not already exist.
        # if it is first in eval order, the non-existent results are overwritten
        Metrics('exists_not', float_acc=3, last_k=None, avg=True, min=True, not_available_value="N/A", name="test/accuracy"),
    ]

    # 5. update the groups
    gm.update_groups(path, metrics, force=False)

    # 6. print nice tables
    print()
    print('6', '-'*100)
    gm.print_csv_table(sort_by='random/r avg', descending=True)
    print('-'*100)
    gm.print_latex_table(sort_by='random/r avg', descending=True)

    # 7. maybe filter your groups, or ignore keys in the table
    print()
    print('7', '-'*100)
    gm2 = gm.copy().filter_groups(dict(model=['resnet18']))
    gm2.print_latex_table(sort_by='random/r avg', descending=True, ignore_keys=['model', 'random/r std'])
