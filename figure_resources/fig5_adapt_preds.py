import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore, truncnorm, sem
import pickle
from functions import regression_indices, convert_to_deg
import os

# get the input regarding which model's features to adapt
parser = argparse.ArgumentParser(
    prog='fig5_adapt_preds.py',
    description='which model you want to adapt features of and get responses from adapted features',
    epilog='only one argument, which model?'
)

parser.add_argument('--model', type=str, required=True, # the argument that specifies the model 
    choices=['resnet18', 'alexnet', 'vgg16', 'vitl32', 'simclr_resnet50'], 
    help='The model to run: resnet18, alexnet, vgg16, vitl32, simclr_resnet50')

parser.add_argument('--outpath', type=str, required=True, # the argument that specifies the output path 
    help='Where would you like your output to be stored?')

args = parser.parse_args()

model_name = args.model 

output_path = args.outpath

# load the tau values that we need to get to create the distribution from which we are drawing
t_vals = np.load('data/tau_arr_s34_good_r2.npy')

# load the model feature responses and regressors
features = np.load(f'data/s5_{model_name}.npy') # need these for adapting and decoding
with open(f'data/{model_name}_xpos_decode_ecc.pickle','rb') as filex:
    xpos_regs = pickle.load(filex)
with open(f'data/{model_name}_ypos_decode_ecc.pickle','rb') as filey:
    ypos_regs = pickle.load(filey)

# preprocess the features by zscoring and applying min-max scaling
features = zscore(features, axis=1)
features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

# randomly assign a T value to each feature for 30 iterations, pulled from the distribution of T that fit neural responses well for S3 and S4
n_iterations = 30
n_feats = features.shape[1]
np.random.seed(seed=42)
feats_ts3 = truncnorm.rvs(a=-2, b=2, loc=np.nanmean(t_vals[:,0]), scale=np.nanstd(t_vals[:,0]), size=(n_feats,n_iterations))
feats_ts4 = truncnorm.rvs(a=-2, b=2, loc=np.nanmean(t_vals[:,1]), scale=np.nanstd(t_vals[:,1]), size=(n_feats,n_iterations))

# define the exponential decay function for a time value x, constant C, and tau value T
def adapt_resp_decay(x, C, T):
    return C*np.exp(-x/T)

# set up time values for adapting the features 
# - keep time 0 as a sanity check that everything is working as it should 
# (i.e., the first time point still has the same correlation as the un-adapted)
times = np.arange(0,3050,50)

# function for adapting and making predictions
def seasons_preds(feats, season_taus, xpos_regressors, ypos_regressors, n_img=40, n_folds=4, n_reps=100, n_time_steps=61):
    xpos_all_pred_array = np.empty((n_img,n_reps,n_time_steps,n_iterations))
    ypos_all_pred_array = np.empty((n_img,n_reps,n_time_steps,n_iterations))
    avg_resp = np.empty((n_time_steps,n_iterations)) # make an array to get the avg response for adaptation at each time and iteration to keep the granularity in case necessary
    avg_resp_sem = np.empty((n_time_steps,n_iterations)) 
    for i in range(n_iterations):
        tmp_tau = season_taus[:,i] # the tau values for this iteration
        for t in range(n_time_steps):
            tmp_feats = adapt_resp_decay(x=times[t],C=feats,T=tmp_tau) # adapting the features up to this time
            avg_resp[t,i] = np.nanmean(np.nanmean(tmp_feats, axis=0))
            avg_resp_sem[t,i] = np.nanmean(sem(tmp_feats, axis=0)) # the mean standard error (of the mean) across images
            for r in range(n_reps):
                for f in range(n_folds):
                    test = regression_indices(n_img, number_of_groups=n_folds, test_group=f, SEED=r)[1]
                    x_test = tmp_feats[test,:]
                    y_pred_xpos = xpos_regressors[f"reg_{r}_{f}"].predict(x_test)
                    y_pred_ypos = ypos_regressors[f"reg_{r}_{f}"].predict(x_test)
                    xpos_all_pred_array[test,r,t,i] = np.squeeze(y_pred_xpos)
                    ypos_all_pred_array[test,r,t,i] = np.squeeze(y_pred_ypos)
    return avg_resp, avg_resp_sem, xpos_all_pred_array, ypos_all_pred_array

# make sure that there is a folder for the adapted results of the model 
if not os.path.exists(f'{output_path}'):
    os.makedirs(f'{output_path}')

# get the position predictions following adaptation to S3
s3_avg_resp, s3_avg_resp_sem, s3_xpos_preds, s3_ypos_preds = seasons_preds(features,feats_ts3,xpos_regs,ypos_regs)

np.save(f'{ouput_path}/{model_name}_s3adapt_avg_resp_ecc.npy', s3_avg_resp)
np.save(f'{output_path}/{model_name}_s3adapt_avg_resp_sem_ecc.npy', s3_avg_resp_sem)
np.save(f'{output_path}/{model_name}_s3adapt_xpos_preds_ecc.npy', s3_xpos_preds)
np.save(f'{output_path}/{model_name}_s3adapt_ypos_preds_ecc.npy', s3_ypos_preds)

# get thr position predictions following adaptation to S4
s4_avg_resp, s4_avg_resp_sem, s4_xpos_preds, s4_ypos_preds = seasons_preds(features,feats_ts4,xpos_regs,ypos_regs)

np.save(f'{output_path}/{model_name}_s4adapt_avg_resp_ecc.npy', s4_avg_resp)
np.save(f'{output_path}/{model_name}_s4adapt_avg_resp_sem_ecc.npy', s4_avg_resp_sem)
np.save(f'{output_path}/{model_name}_s4adapt_xpos_preds_ecc.npy', s4_xpos_preds)
np.save(f'{output_path}/{model_name}_s4adapt_ypos_preds_ecc.npy', s4_ypos_preds)