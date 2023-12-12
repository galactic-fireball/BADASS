#!/bin/sh

#SBATCH --job-name=BADASS_UNIT_TESTS

#SBATCH --partition=normal # HOPPER
#SBATCH --output=/scratch/%u/%x-%j.out # user, jobName, jobID
#SBATCH --error=/scratch/%u/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@gmu.edu

#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --mem=30G


module load python/3.8.6-ff # HOPPER
source /home/sdoan2/python-ve/env/bin/activate

python -m unittest ../../testing/run_tests.py
