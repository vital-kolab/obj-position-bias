# Calculate x-position prediction deltas (S3 vs S5, S4 vs S5, S3 vs S4) as a function of the number of neurons in the regression

# imports 
import argparse
import h5py
import pandas as pd
import numpy as np
from scipy.stats import zscore, shapiro, levene
from sklearn.linear_model import Ridge, RidgeCV

import sys
sys.path.append('../util_code')

from functions import reliability_filtering, regression_indices, convert_to_deg

# functions
def train_regs(neural, pos_vals, n_img=40, n_runs=100, n_folds=4, alpha_values = [0.01, 0.1, 1, 10, 100]):
    '''
    Function to train regressors to predict position from neural responses.
    neural = neural data, numpy array, [n_img,n_neurons]
    pos_vals = position values, numpy array, [n_img]
    n_img = number of images in the dataset, int
    n_runs = number of shuffles of fold group assignment, int
    n_folds = number of groups in the train and test split, int
    alpha_values = alpha values to be tested by the regressor, int
    '''
    regressors = {}
    preds = np.empty((n_img,n_runs))
    for x in range(n_runs):
        for i in range(n_folds):
            train, test = regression_indices(n_img, number_of_groups=4, test_group=i, SEED=x)
            x_train = neural[train,:]
            y_train = pos_vals[train]
            x_test = neural[test,:]
            ridge_cv = RidgeCV(alphas=alpha_values).fit(x_train,y_train)
            best_alpha = ridge_cv.alpha_
            regressors[f"reg_{x}_{i}"] = Ridge(alpha=best_alpha) 
            regressors[f"reg_{x}_{i}"].fit(x_train,y_train)
            y_pred = regressors[f"reg_{x}_{i}"].predict(x_test)
            preds[test, x] = np.squeeze(y_pred)
    preds = np.nanmean(preds,axis=1)
    return regressors, preds

def test_regs(neural, regressors, n_img=40, n_runs=100, n_folds=4):
    '''
    neural = neural data, numpy array, [n_img,n_neurons]
    regressors = the trained linear regression models that predict position from neural responses, dict, indexed with reg_x_i
    n_img = number of images in the dataset, int
    n_runs = number of shuffles of fold group assignment, int
    n_folds = number of groups in the train and test split, int
    '''
    preds = np.empty((n_img,n_runs))
    for x in range(n_runs):
        for i in range(n_folds):
            test = regression_indices(n_img, number_of_groups=4, test_group=i, SEED=x)[1]
            x_test = neural[test,:]
            y_pred = regressors[f"reg_{x}_{i}"].predict(x_test)
            preds[test,x] = np.squeeze(y_pred)
    preds = np.nanmean(preds,axis=1)
    return preds

# parallelization tools
parser = argparse.ArgumentParser(
    prog='fig3_deltas.py',
    description='specify start and stop of the range of neurons this script will work with',
    epilog='two arguments: start and stop, both integers'
)

parser.add_argument('--start', type=int, required=True) # the argument that specifies the number of subsamples to test 
parser.add_argument('--stop', type=int, required=True)
parser.add_argument('--outpath', type=str, required=True, # the argument that specifies the output path 
    help='Where would you like your output to be stored?')

args = parser.parse_args()

n_start = args.start  
n_stop = int(args.stop + 10) # to ensure that range, which is end exclusive, actually gets to the number of neurons we are looking for
n_subs = int((n_stop - n_start)/10)

output_path = args.outpath

# import the neural data 
m1_file_s5 = h5py.File('../data/neural/230801.m1.rsvp.gratingsAdap_s5.experiment_psth_raw.h5','r')
m2_file_s3 = h5py.File('../data/neural/230801.m1.rsvp.gratingsAdap_s3.experiment_psth_raw.h5','r')
m1_file_s4 = h5py.File('../data/neural/230801.m1.rsvp.gratingsAdap_s4.experiment_psth_raw.h5','r')

m1_rates_s5 = m1_file_s5['/psth'][:]
m1_rates_s3 = m1_file_s3['/psth'][:]
m1_rates_s4 = m1_file_s4['/psth'][:]
m2_rates_s5 = np.load('../data/neural/m2_season5.npy')
m2_rates_s3 = np.load('../data/neural/m2_season3.npy')
m2_rates_s4 = np.load('../data/neural/m2_season4.npy')

# process M1 neural data 
s5_neural_m1 = np.transpose(np.nanmean(m1_rates_s5[:,:,7:17,:],axis=2), (0,2,1))
s3_neural_m1 = np.transpose(np.nanmean(m1_rates_s3[:,:,337:347,:],axis=2), (0,2,1)) 
s4_neural_m1 = np.transpose(np.nanmean(m1_rates_s4[:,:,337:347,:],axis=2), (0,2,1)) 

# concatenate and filter the combined neural data 
rel = np.load('../data/combined_s5_rel.npy')

s5_neural = reliability_filtering(np.concatenate((s5_neural_m1,m2_rates_s5),axis=1),rel,metric=0.2)
s3_neural = reliability_filtering(np.concatenate((s3_neural_m1,m2_rates_s3),axis=1),rel,metric=0.2)
s4_neural = reliability_filtering(np.concatenate((s4_neural_m1,m2_rates_s4),axis=1),rel,metric=0.2)

# zscore the data and process for making predictions
s5_neural = zscore(s5_neural,axis=1)
s3_neural = zscore(s3_neural,axis=1)
s4_neural = zscore(s4_neural,axis=1)

s5_neural_for_preds = np.nanmean(s5_neural,axis=2)
s3_neural_for_preds = np.nanmean(s3_neural,axis=2)
s4_neural_for_preds = np.nanmean(s4_neural,axis=2)

# load the position data (pixel space)
human_im_size = 268.51895786308
gt_im_size = 256
df = pd.read_csv('../data/mae_s5_coordinates.csv') # position data
pos = np.squeeze(df['center_x'].values.reshape(-1,1))*(human_im_size/gt_im_size) # convert to the dimensions of the image that the humans made estimates on
pos = convert_to_deg(pos, human_im_size) # convert labels to eccentricity

# build the loop behaviour
n_reps = 20 # 20 repetitions of subsampling at each number of neurons
global_seed = 42 # set a global seed for reproducibility
s35_global_store = np.empty((n_subs,n_reps,40)) # store the final imagewise deltas averaged across all repetitions 
s45_global_store = np.empty((n_subs,n_reps,40))
s43_global_store = np.empty((n_subs,n_reps,40))
s = 0 # the number of subsamples we are at
for n in range(n_start,n_stop,10): # the values at which to start and stop subsampling are defined in the input
    s35_store = np.empty((n_reps,40)) # store the imagewise deltas for each repetition of each subsample
    s45_store = np.empty((n_reps,40))
    s43_store = np.empty((n_reps,40))
    for r in range(n_reps):
        its = (global_seed * 1000 + n * 100 + r) # a unique seed for every set of nested iterations 
        rng = np.random.default_rng(seed=its)
        units_idx = rng.choice(s5_neural_for_preds.shape[1], size=n, replace=False)
        s5_tmp = s5_neural_for_preds[:,units_idx]
        s3_tmp = s3_neural_for_preds[:,units_idx]
        s4_tmp = s4_neural_for_preds[:,units_idx]
        s5_regressors, s5_preds = train_regs(s5_tmp,pos)
        s3_preds = test_regs(s3_tmp,s5_regressors)
        s4_preds = test_regs(s4_tmp,s5_regressors)
        s35_store[r,:] = s3_preds-s5_preds
        s45_store[r,:] = s4_preds-s5_preds
        s43_store[r,:] = s4_preds-s3_preds
    s35_global_store[s,:,:] = s35_store
    s45_global_store[s,:,:] = s45_store
    s43_global_store[s,:,:] = s43_store
    s += 1

# output to scratch - filenames edited for eccentricity version
np.save(f'{output_path}/ds35x_{n_start}-{args.stop}_ecc_reps.npy',s35_global_store)
np.save(f'{output_path}/ds45x_{n_start}-{args.stop}_ecc_reps.npy',s45_global_store)
np.save(f'{output_path}/ds43x_{n_start}-{args.stop}_ecc_reps.npy',s43_global_store)