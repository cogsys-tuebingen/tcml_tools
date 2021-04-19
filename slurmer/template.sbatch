#!/bin/bash

#SBATCH --job-name={job_name}

#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --gres=gpu:1080ti:{gpus}

#SBATCH --partition={partition}
#SBATCH --time={time}
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error={out_path}{name}.err
#SBATCH --output={out_path}{name}.out

#SBATCH --mail-type={mail_type}
#SBATCH --mail-user={user_mail}


srun -N1 -n1 -c $SLURM_CPUS_PER_TASK --gres=gpu:1080ti:{gpus} singularity exec --nv {img} {path}
sleep 10
echo DONE!
