#!/bin/bash
#BSUB -n 1
#BSUB -W 20
#BSUB -q gpu
#BSUB -R "select[a30]"
#BSUB -gpu "num=1"
#BSUB -o out.%J
#BSUB -e err.%J
module load matlab
matlab -nodisplay -nosplash -nodesktop -singleCompThread -r "run('matlab_gpu_benchmark.m');exit;"
