"""
execute /run_on_cluster/log_tb.py on the cluster with some arbitrary parameters

TODO:
- set email
- set username
- run on the cluster:
    - ssh to the cluster
    - set the pythonpath: export PYTHONPATH=${PYTHONPATH}:$HOME/code/tcml_tools/
    - cd to the script and run: python3 log_tb1.py
- copy the 4 "gm.add_group(...)" output lines for the next file
"""

from slurmer.slurmer import Slurmer
from slurmer.queues import RunType


if __name__ == "__main__":
    slurmer_ = Slurmer(paste_path='~/experiments/_scripts/')

    # you could set partition/time strings yourself, but you can also easily set it like this, enables quick testing
    run_type = RunType.TEST  # FULL, TEST
    partition, time = run_type.get_time_partition(test_minutes=5, hours=1, minutes=10)  # set required time here
    print("using partition '%s' for time '%s', is a test run: %s" % (partition, time, run_type.is_test()))

    # pure string replacements for the templates
    slurmer_.set_str_replacements({
        # the name shows up when running squeue
        'job_name': 'test',

        # partition and time
        'partition': partition,
        'time': time,

        # email
        'user_mail': '<my-email>@uni-tuebingen.de',  # TODO set your email
        'mail_type': 'ALL',

        # resources
        'cpus_per_task': '4',
        'mem_per_cpu': '12G',
        'gpus': '1',

        # singularity image to run your code in
        'img': '/common/singularityImages/TCML-Cuda11_0_TF2_4_1_PT1_7_1.simg',

        # where to dump the .err and .out files
        # sadly ${HOME} or ${SLURM_JOB_ID} do not work here
        # local paths (./scripts/...) do work, if you do not mind having them there
        'out_path': "/home/<my-username>/experiments/_outfiles/{project}/{type}/%s/" % partition,  # TODO set user name

        # generic replacements, use e.g. for data/save dirs
        'code_dir': '${HOME}/code/',
        'project': 'tcml_tools',
        'file': 'examples/on_cluster/log_tb.py',
        'type': 'log_tb'
        # 'name' is automatically generated from used grid search values of each particular job
    })

    # shared (therefore fixed) values for argparse in the python file, we can use the above set replacements here
    slurmer_.set_sh_shared_values({
        'data_dir': '"/scratch/${SLURM_JOB_ID}/"',
        'save_dir': '"${HOME}/experiments/{project}/{type}/%s/${SLURM_JOB_ID}/{name}/"' % partition,
        'epochs': 200,
        'lr': 0.1,
    })

    # grid search values for argparse in the python file
    slurmer_.set_param_values([
        ('seed', [0, 1]),
        ('mu', [0.1, 0.4]),
        ('model', ['vgg16', 'resnet18']),
    ])

    # copy the data set and add the python path in the sh template, replacing {lines_before} and {lines_after}
    # good place to check run_type.is_test() if do not need the full data set for quick test runs
    slurmer_.add_sh_line('cp -R /common/datasets/cifar10/cifar-10-batches-py/ /scratch/${SLURM_JOB_ID}/', before=True)
    slurmer_.add_sh_line('export PYTHONPATH=${PYTHONPATH}:${HOME}/{code_dir}/{project}/', before=True)
    slurmer_.add_sh_line('sleep 5', before=False, after=True)

    # now run everything
    slurmer_.create_files(verbose=True)
    slurmer_.queue_files()
    slurmer_.print_table()
    slurmer_.print_group_code(ignore=('seed',))  # group up all jobs that differ only in seed
