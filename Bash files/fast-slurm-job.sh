#!/bin/bash
#
#SBATCH --job-name=zip_results
#SBATCH --output=ziplog.out
#
#SBATCH --ntasks=1
#SBATCH --time=8-
#SBATCH --cpus=4
#SBATCH --mem-per-cpu=4000
zip -r E763-790.zip output logs 
