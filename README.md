
# Tools for the TCML Slurm Cluster

For infos about the cluster, check
[this link](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/projects/tcml-cluster/)

Be responsible about your usage of the cluster.
Since the resources are shared and limited, bad behavior may necessitate administrative actions.


----

## standard slurm workflow

Generally you need to create a .sbatch script, which makes slurm run some python script in a
singularity container.
Slurm will create error and output files from the respective console streams and e-mail you
the status of the job.

In many cases you want to have slurm execute a shell script instead, so you can also e.g. copy
a dataset or set the python path before. So generally you will be working with two files already.

If you now also want to run a script multiple times with e.g. different seeds, you must not
modify your shell script until slurm started the job, or you need a separate one.
That can quickly get out of hand.

----

## an easier way

The Slurmer provided here automatically creates sbatch+shell scripts and queues them,
while you still have the control over what is happening.


### set up

Copy/Clone the project and move it to your home dir on the cluster.

The expected path for the examples is: ${HOME}/code/tcml_tools/


### check that it works as expected

ssh to the cluster and cd to /tcml_tools/examples/

Follow the instructions in log_tb1.py,
wait briefly until all jobs are finished,
then follow the instructions in log_tb2.py.

----


## TODO in the future

- Check out [submitit](https://github.com/facebookincubator/submitit),
  which was published after I already wrote most of this code.
  The use case seems to differ though.
