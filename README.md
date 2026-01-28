# The macaque IT cortex but not current artificial vision networks encode object position in perceptually aligned coordinates.

This repo contains the codebase accompanying the paper: 

**The macaque IT cortex but not current artificial vision networks encode object position in perceptually aligned coordinates**, Yakubovskaya et al., *Current Biology*, 2026 (INSERT LINK HERE)

This repository supports analyses comparing the effect of motion adaptation on estimates of object position by humans, readout from macaque inferior temporal (IT) cortex, and predictions from artificial neural networks (ANNs).

## 🗂️ Respository layout
* **data/** <br> This folder contains the neural data, human behavioural data, and ANN features you will need to run all the code in this repository. It can be downloaded from our OSF page here (INSERT LINK HERE).
  
* **figure_notebooks/** <br> This folder contains the notebooks to generate all the figures in the paper. All the data and intermediates are provided in **data/**.

* **figure_resources/** <br> Some figures require intermediate analyses that were not conducted in a notebook, the Python and Bash scripts for such analyses are contained here. However, all the results of these analyses are provided in **data/**, and as such, the code in this folder is for your interest in replicating some of the intermediate results of our paper.

* **util_code/** <br> This folder contains functions required throughout the repository, as well as the files necessary to set up your environment.

## 🔨 Setup 

We recommend you begin by cloning this repository:

```
# Clone the repository
git clone https://github.com/vital-kolab/obj-position-bias.git
cd obj-position-bias
```

Please edit `util_code/setup.sh` and `util_code/requirements.txt` to fit your system, then set up a fresh environment (Conda or venv) with Python ≥3.10 (we used Python 3.11.5):

```
# Set up your environment
cd util_code
setup.sh
```

## 📀 Data and Preparation

This project assumes access to the data stored here (INSERT LINK HERE). For best compatibility, please store the **data** folder within this repository (i.e., `../obj-position-bias/data`).

Within **data** you will have access to:

  1. **Macaque IT responses**: population responses for N images.
     * `/neural` shape for m1 `(n_images, n_reps, n_time, n_neurons)` and for m2 `(n_images, n_neurons, n_reps)`
  3. **Model features**: precomputed ANN activations for the same images.
     * `/convrnn_features` shape `(n_images, n_units)`
     * `/slowfast_features` shape `(n_images, n_units, n_time, H, W)`
     * `/static_img_model_features` shape `(n_images, n_units)`
  5. **Human behavior**: image-level object position estimates.
     * `/human` shape `(n_images, n_trials)`

## 🔁 Reproducing paper figures

Each `/figure_notebooks/figureX.ipynb` notebook reproduces the corresponding figure from the paper. Notebooks expect the data assets described above. 

## 📌 Status



