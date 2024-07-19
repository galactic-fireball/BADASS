#!/bin/sh

#SBATCH --job-name=BADASS_UNIT_TESTS

#SBATCH --partition=normal
#SBATCH --output=/scratch/%u/%x-%j.out # user, jobName, jobID
#SBATCH --error=/scratch/%u/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@gmu.edu

#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --mem=30G

module load gnu10/10.3.0-ya
module load python/3.10.1-qb
source /home/sdoan2/venv/bin/activate

python -m unittest testing/run_tests.py
