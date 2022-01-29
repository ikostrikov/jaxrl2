#!/bin/bash
#SBATCH --job-name=sac
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TITAN:1
#SBATCH --account=co_rail
#SBATCH --partition=savio3_gpu
#SBATCH --qos=rail_gpu3_normal


ENV_ID=$((SLURM_ARRAY_TASK_ID-1))

arrENVS=(${ENVS//;/ })

ENV_NAME=${arrENVS[$ENV_ID]}

module load gnu-parallel

echo $ENV_NAME

run_singularity ()
{
    singularity exec --nv --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay $SCRATCH/singularity/overlay-50G-10M.ext3:ro $SCRATCH/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

    source ~/.bashrc

    XLA_PYTHON_CLIENT_PREALLOCATE=false python ../examples/train_online.py --env_name=$2 \
                    --seed=$1 \
                    --save_dir=logs/results/sac_target/$2/ \
                    --config=../examples/configs/sac_default.py \
                    --use_actor_target \
                    --notqdm
    "
}
export -f run_singularity

parallel --delay 20 --linebuffer -j 4 run_singularity {} $ENV_NAME ::: 1 2 3 4 5 6 7 8 9 10