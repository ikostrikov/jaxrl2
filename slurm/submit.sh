#!/bin/bash

# ENVS="walker-walk;cheetah-run;ball_in_cup-catch;cartpole-swingup;finger-spin;reacher-easy"
ENVS="HalfCheetah-v3;Hopper-v3;Walker2d-v3;Ant-v3;Humanoid-v3"

mkdir logs/out/ -p
mkdir logs/err/ -p

arrENVS=(${ENVS//;/ })
NUM_ENVS=${#arrENVS[@]}

sbatch --export=ENVS=$ENVS,SEEDS=$SEEDS,OUTDIR=$OUTDIR,ERRDIR=$ERRDIR --array=1-${NUM_ENVS}%10 sbatch.sh