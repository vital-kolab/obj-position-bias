#!/bin/bash

module load python/3.11.5 scipy-stack hdf5/1.14.2

WORKING_PATH="/home/eyakub/projects/def-kohitij/eyakub/obj-position-bias"

echo "Start Installing and setup env"

python -m venv $WORKING_PATH/.venv

source $WORKING_PATH/.venv/bin/activate

echo "Installing requirements"

pip install --no-index -r $WORKING_PATH/requirements.txt

echo "Env has been set up"

pip freeze

