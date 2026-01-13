#!/bin/bash
#SBATCH --job-name=zoo_preds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=your.email.here@email.com
#SBATCH --mail-type=ALL

# NOTE: this job script is intended for Compute Canada, please adapt above and below to your system 
WORKING_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias" # adjust for your system
ENV_PATH="/home/eyakub/scratch/.obj_pos_venv" # adjust for your system
OUTPUT_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias/data"

# navigate to the location this is being run and make log directory
cd $WORKING_PATH
mkdir -p logs

# set up the node 
module --force purge # intended to work with compute canada, adjust for your system
module load python/3.11.5 

# make the environment
source $ENV_PATH/bin/activate
echo "Env has been activated"
pip freeze

# run the actual python script
python zoo_preds.py --outpath $OUTPUT_PATH