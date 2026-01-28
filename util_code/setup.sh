#!/bin/bash

module load python/3.11.5 # intended to work with a Digital Research Alliance of Canada system and should be adjust for the capabilities of your system

ENV_PATH="/home/eyakub/scratch" # adjust for your system
WORKING_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias" # adjust for your system

echo "Start Installing and setup env"

python -m venv $ENV_PATH/.obj_pos_venv

source $ENV_PATH/.obj_pos_venv/bin/activate

echo "Installing requirements"

# NOTE: this code was written for and run on a Digital Research Alliance of Canada system, if your system is not part of this network, remove '+computecanada' from requirements.txt
pip install --no-index -r $WORKING_PATH/util_code/requirements.txt 

echo "Env has been set up"

pip freeze

echo "Making the environment a kernel"

python -m ipykernel install \
  --user \
  --name obj-position-bias \
  --display-name "Python (obj-position-bias)"

echo "Setup complete!"
