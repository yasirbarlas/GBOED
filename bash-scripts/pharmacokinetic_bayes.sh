#!/bin/bash
#SBATCH --job-name pharm_bayes
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

seeds=($(seq 50 50 10000))
for id in "${seeds[@]}"
do
   python3 pharmacokinetic.py --seed=$id --T=5 --N=10000 --M=100 --name="" --typs="nmc" --inference="bayesian" --lengthscale=20.0 --variance=10.0 --num-acquisition=500 --assumed-epsilon-scale=0.01 --assumed-nu-scale=0.1 --w=0.4 --chosen-loss="neg-log" --misspecification="none" --actual-epsilon-scale=0.01 --actual-nu-scale=0.1 --c-imq=0.0 --b-use-expdecay=False --b-expdecay-imq=0.0
done
