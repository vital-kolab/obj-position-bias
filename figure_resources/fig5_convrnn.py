# For Figure 5 D
# training decoders ConvRNN responses to images before motion adaptation and making 
# predictions using responses to images following right (s3) or left (s4) motion adaptation

# imports
import argparse
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
import pandas as pd
from scipy.stats import zscore
import pickle

import sys
sys.path.append('../util_code')

from functions import regression_indices, convert_to_deg

# get output_path
parser = argparse.ArgumentParser( # make the parser
    prog='fig5_convrnn.py', 
    description='specify output path', 
    epilog='only one argument, what is your output path') 

parser.add_argument('--outpath', type=str, required=True, # the argument that specifies the output path 
    help='Where would you like your output to be stored?')

args = parser.parse_args()

output_path = args.outpath

# define a function to prepare the features for making predictions
def prep_feats(features):
    features = zscore(features, axis=1)
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    return features

# define the function to train the regressor
def train_regressors(feats, pos, n_img = 40, n_folds = 4, n_reps = 100, alpha_values = [0.01, 0.1, 1, 10, 100]):
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

# define the function that gets predictions for s3 and s4
def seasons_preds(feats, xpos_regressors, ypos_regressors, n_img=40, n_folds=4, n_reps=100):
    xpos_all_pred_array = np.empty((n_img,n_reps))
    ypos_all_pred_array = np.empty((n_img,n_reps))
    for r in range(n_reps):
        for f in range(n_folds):
            test = regression_indices(n_img, number_of_groups=n_folds, test_group=f, SEED=r)[1]
            x_test = feats[test,:]
            y_pred_xpos = xpos_regressors[f"reg_{r}_{f}"].predict(x_test)
            y_pred_ypos = ypos_regressors[f"reg_{r}_{f}"].predict(x_test)
            xpos_all_pred_array[test,r] = np.squeeze(y_pred_xpos)
            ypos_all_pred_array[test,r] = np.squeeze(y_pred_ypos)
    return xpos_all_pred_array, ypos_all_pred_array


# load and prepare the features 
s5 = prep_feats(np.load('../data/convrnn_features/s5_convrnn.npy'))
s3 = prep_feats(np.load('../data/convrnn_features/s3_convrnn.npy'))
s4 = prep_feats(np.load('../data/convrnn_features/s4_convrnn.npy'))

# load the ground truth center positions in pixel space in ecc coordinates
human_im_size = 268.51895786308
gt_im_size = 256
df = pd.read_csv('../data/mae_s5_coordinates.csv') # position data
xpos = np.squeeze(df['center_x'].values.reshape(-1,1))*(human_im_size/gt_im_size) # convert to the dimensions of the image that the humans made estimates on
xpos = convert_to_deg(xpos,human_im_size)
ypos = np.squeeze(df['center_y'].values.reshape(-1,1))*(human_im_size/gt_im_size)
ypos = convert_to_deg(ypos,human_im_size)

# get s5 regressors and predictions for x pos
xpos_regs, xpos_preds = train_regressors(s5, xpos)

# output the predictions for x-position
np.save(f'{output_path}/convrnn_s5_xpos_preds_ecc.npy',xpos_preds)

# get s5 regressors and predictions for y pos
ypos_regs, ypos_preds = train_regressors(s5, ypos)

# output regressors and predictions for y 
np.save(f'{output_path}/convrnn_s5_ypos_preds_ecc.npy',ypos_preds)

# get predictions from s3 and s4 features 
s3_xpos_preds, s3_ypos_preds = seasons_preds(s3,xpos_regs,ypos_regs)
np.save(f'{output_path}/convrnn_s3_xpos_preds_ecc.npy', s3_xpos_preds)
np.save(f'{output_path}/convrnn_s3_ypos_preds_ecc.npy', s3_ypos_preds)

s4_xpos_preds, s4_ypos_preds = seasons_preds(s4,xpos_regs,ypos_regs)
np.save(f'{output_path}/convrnn_s4_xpos_preds_ecc.npy', s4_xpos_preds)
np.save(f'{output_path}/convrnn_s4_ypos_preds_ecc.npy', s4_ypos_preds)