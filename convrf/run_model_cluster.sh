#!/bin/bash
#SBATCH -J test
#SBATCH -o test.o%j
#SBATCH -t 80:00:00
#SBATCH -N 1 -n 8
#SBATCH -A  labate
#SBATCH  -p gpu
#SBATCH  --gres=gpu:1

if [ "$1" != "" ]; then
    echo "$1"
else
    echo "Positional parameter 1 should be the output folder"
fi
module load Anaconda3
python3 example.py $1
