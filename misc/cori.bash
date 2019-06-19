#!/bin/bash
#SBATCH -J QuantumSearchCompiler
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --time-min=02:0:00
#SBATCH --comment=168:00:00
#SBATCH --signal=B:USR1@10
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --error=qsc-%j.err
#SBATCH --output=qsc-%j.out
#SBATCH --mail-user=ethanhs@nersc.gov

max_timelimit=48:00:00
ckpt_overhead=10
ckpt_command=

. /usr/common/software/variable-time-job/setup.sh<Paste>
requeue_job func_trap USR1

srun -c32 --cpu_bind=cores ./qsc &

wait


