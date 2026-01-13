# For Figure 1 D

# script that will train decoders to predict position of objects in HVM640 images based on model responses
# and save predictions

import argparse 
import os

# get inputs
parser = argparse.ArgumentParser( # make the parser
    prog='fig1_preds.py', 
    description='choose which model you want each node to run', 
    epilog='only one argument, which model you want?') 

parser.add_argument('--model', type=str, required=True, # the argument that specifies the model 
    choices=['resnet18', 'alexnet', 'vgg16', 'vitl32', 'simclr_resnet50','resnet50_ssl','resnet50_robust_eps3'], 
    help='The model to run: resnet18, alexnet, vgg16, vitl32, simclr_resnet50, resnet50_ssl, resnet50_robust_eps3')

parser.add_argument('--outpath', type=str, required=True, # the argument that specifies the model 
    help='Where would you like your output to be stored?')

args = parser.parse_args()

model_key = args.model
output_path = args.outpath

# imports 
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import zscore
import pickle

from functions import regression_indices

# define the function to train the regressors and get the dictionary of trained regressors as well as predictions
def train_regressors(feats, pos, n_img = 640, n_folds = 10, n_reps = 100, alpha_values = [0.01, 0.1, 1, 10, 100]):
    regressors = {}
    pred_array = np.empty((n_img,n_reps))

    for r in range(n_reps):
        for f in range(n_folds):
            train, test = regression_indices(n_img, number_of_groups=n_folds, test_group=f, SEED=r)
            x_train = feats[train,:]
            y_train = pos[train]
            x_test = feats[test,:]
            ridge_cv = RidgeCV(alphas=alpha_values).fit(x_train,y_train) # performs leave-one-out cross validated alpha optimization
            best_alpha = ridge_cv.alpha_
            regressors[f"reg_{r}_{f}"] = Ridge(alpha=best_alpha)
            regressors[f"reg_{r}_{f}"].fit(x_train,y_train)
            y_pred = regressors[f"reg_{r}_{f}"].predict(x_test)
            pred_array[test,r] = np.squeeze(y_pred)
    
    return regressors, pred_array

# load the ground truth center positions
df = pd.read_csv('data/meta_hvm640.csv') # position data
xpos = np.squeeze(df['xpos'].values.reshape(-1,1))
ypos = np.squeeze(df['ypos'].values.reshape(-1,1))

features = np.load(f'data/hvm640_{model_key}.npy')

# preprocess the features by zscoring and applying min-max scaling
features = zscore(features, axis=1)
features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

# get the regressors for x position
xpos_regs, xpos_preds = train_regressors(features,xpos)

# output the regressors and the predictions for x
with open(f'{output_path}/{model_key}_xpos_decode.pickle', 'wb') as handlex:
    pickle.dump(xpos_regs, handlex, protocol=pickle.HIGHEST_PROTOCOL)

np.save(f'{output_path}/{model_key}_hvm640_xpos_preds',xpos_preds)

# get the regressors for y position
ypos_regs, ypos_preds = train_regressors(features,ypos)

# output the regressors and the predictions for y
with open(f'{output_path}/{model_key}_ypos_decode.pickle', 'wb') as handley:
    pickle.dump(ypos_regs, handley, protocol=pickle.HIGHEST_PROTOCOL)

np.save(f'{output_path}/{model_key}_hvm640_ypos_preds',ypos_preds)