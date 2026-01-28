#!/bin/bash
#SBATCH --job-name=fig3_deltas
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=15G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=your.email.here@email.com
#SBATCH --mail-type=ALL

# NOTE: this job script is intended for Compute Canada, please adapt above and below to your system 

# define paths
WORKING_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias/figure_resources" # adjust for your system
ENV_PATH="/home/eyakub/scratch/.obj_pos_venv" # adjust for your system
OUTPUT_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias/data"

# navigate to the location this is being run and make log directory
cd $WORKING_PATH
mkdir -p logs

# set up the node 
module --force purge # intended to work with compute canada, adjust for your system
module load python/3.11.5 

# activate the environment
source $ENV_PATH/bin/activate
echo "Env has been activated"
pip freeze

# run the script
srun --exclusive -N1 -n1 python fig3_deltas.py  --start 10 --stop 50 --outpath $OUTPUT_PATH &
srun --exclusive -N1 -n1 python fig3_deltas.py --start 60 --stop 110 --outpath $OUTPUT_PATH &

wait