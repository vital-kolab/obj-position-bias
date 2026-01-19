#!/bin/bash
#SBATCH --job-name=zoo_adapt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=eyakub@my.yorku.ca
#SBATCH --mail-type=ALL

# navigate to the location this is being run and make log directory
cd ~/projects/def-kohitij/eyakub/decoder_sandbox
mkdir -p logs

# load the node
module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack hdf5/1.14.2

# make the environment
pip freeze
virtualenv --no-download $SLURM_TMPDIR/env
pip install --no-index --upgrade pip
echo "Installing requirements"
pip install --no-index -r requirements.txt 
# source ~/projects/def-kohitij/eyakub/decoder_sandbox/decode_env/bin/activate
echo "Env has been set up"
pip freeze

# run the different models on different nodes - & makes it so that all the models are run in the background on separate nodes
# srun --exclusive -N1 -n1 python adapt_zoo.py --model resnet18 &
# srun --exclusive -N1 -n1 python adapt_zoo.py --model alexnet &
# srun --exclusive -N1 -n1 python adapt_zoo.py --model vgg16 &
# srun --exclusive -N1 -n1 python adapt_zoo.py --model vitl32 &
# srun --exclusive -N1 -n1 python adapt_zoo.py --model simclr_resnet50 &

# wait #NOTE memory should be increased to 80G if running all the models 

python adapt_zoo.py --model vitl32