#!/bin/bash
#SBATCH --job-name loc_bayes
#SBATCH --partition=fill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=96:00:00
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o
#SBATCH --error err_bayes.err
#SBATCH --output out_bayes.output

module load apps/binapps/anaconda3/2022.10
source activate gen_bayes

python --version

lscpu

#pip freeze
#Run your script.

seeds=($(seq 50 50 5000))
for id in "${seeds[@]}"
do
   python3 location.py --seed=$id --T=30 --N=10000 --M=100 --name="" --typs="nmc" --inference="bayesian" --lengthscale=15.0 --variance=4.0 --num-acquisition=500 --observation-sd=0.5 --w=0.2 --chosen-loss="neg-log" --misspecification="none" --actual-observation-sd=0.5 --c-imq=0.0 --b-use-expdecay=False --b-expdecay-imq=0.0 --k=2 --d=2
done