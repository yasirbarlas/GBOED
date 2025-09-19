#!/bin/bash
#SBATCH --job-name expdec0.04
#SBATCH --partition=fill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=96:00:00
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o
#SBATCH --error err_expdec0.04.err
#SBATCH --output out_expdec0.04.output

module load apps/binapps/anaconda3/2022.10
source activate gen_bayes

python --version

lscpu

#pip freeze
#Run your script.

seeds=($(seq 50 50 1500))
for id in "${seeds[@]}"
do
   python3 regression.py --seed=$id --N=10000 --M=100 --typ="gibbs-nmc" --inference="gibbs" --noise_std=1.2 --w=1.0 --chosen_loss="score-matching-weighted" --true_beta 10.0 -7.0 --misspecification="none" --T=10 --c-imq=0.0 --b-use-expdecay=True --b-expdecay-imq=0.04
   python3 regression.py --seed=$id --N=10000 --M=100 --typ="gibbs-nmc" --inference="gibbs" --noise_std=0.8 --w=1.0 --chosen_loss="score-matching-weighted" --true_beta -3.0 8.0 --misspecification="none" --T=10 --c-imq=0.0 --b-use-expdecay=True --b-expdecay-imq=0.04
   python3 regression.py --seed=$id --N=10000 --M=100 --typ="gibbs-nmc" --inference="gibbs" --noise_std=1.0 --w=1.0 --chosen_loss="score-matching-weighted" --true_beta 9.0 9.0 --misspecification="none" --T=10 --c-imq=0.0 --b-use-expdecay=True --b-expdecay-imq=0.04
done