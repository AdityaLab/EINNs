import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from epiweeks import Week, Year
import pandas as pd
global N
import pdb 

dtype = torch.float

DAY_WEEK_MULTIPLIER = 7
SMOOTH_WINDOW = 21
WEEKS_AHEAD = 8
PAD_VALUE = -9
DAYS_IN_WEEK = 7
# daily
datapath = './data/covid-hospitalization-daily-all-state-merged_vEW202133.csv'
datapath_weekly = './data/covid-hospitalization-all-state-merged_vEW202133.csv'
county_datapath = f'./Data/Processed/county_data.csv'
EW_START_DATA = '202020'  # defaul if not provided
SMOOTH_MOVING_WINDOW = True
population_path = './data/table_population.csv'

# Select signals
macro_features=[
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'apple_mobility',
    'cdc_hospitalized',
    'covidnet',
    'fb_survey_cli',
    'death_jhu_incidence',
    'positiveIncr',
    'negativeIncr',
    ] 

regions = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']

# no calibration for WY, AK, MT
all_hhs_regions = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'X']

# regions = ['CA', 'CT', 'FL', 'KS', 'LA', 'MI', 'MS', 'NV', 'WA', 'X']

# HSS regions https://www.hhs.gov/about/agencies/iea/regional-offices/region-4/index.html
hhs_region4 = ['AL','FL','GA','KY','MS','NC','SC','TN','X'] # HHS 4 + US

hhs_region6 = ['AR','LA','NM','OK','TX'] # HHS 6

hhs_region5 = ['IL','IN','MI','MN','OH','WI'] # HHS 5

hhs_region1 = ['CO','ME','MA','NH','RI','VT']


########################################################
#           helpers
########################################################

def convert_to_epiweek(x):
    return Week.fromstring(str(x))

def load_rmse_data(start_week,pred_week,region):
    df_rmse = pd.read_csv('./data/analytical/{}/RMSE_data.csv'.format(region))
    df_rmse['epiweek'] = df_rmse.loc[:,'epiweek'].apply(convert_to_epiweek)
    df_rmse = df_rmse[(df_rmse['epiweek'] <= convert_to_epiweek(pred_week)) & (df_rmse['epiweek'] >= convert_to_epiweek(start_week))]
    df_rmse.drop(['epiweek'],1,inplace=True) # drop week col
    return df_rmse.to_numpy()  # only

def get_epiweeks_list(start_ew,end_ew):
    """
        returns list of epiweeks objects between start_ew and end_ew (inclusive)
        this is useful for iterating through these weeks
    """
    iter_weeks = list(Year(2020).iterweeks()) + list(Year(2021).iterweeks())
    idx_start = iter_weeks.index(start_ew)
    idx_end = iter_weeks.index(end_ew)
    return iter_weeks[idx_start:idx_end+1]

def create_window_seqs(
        X: np.array, 
        rmse: np.array, 
        y: np.array, 
        y_weights: np.array, 
        min_sequence_length: int,
        pad_value: float=PAD_VALUE
        ):
    """
        Creates windows of fixed size with appended zeros
        @param X: features
        @param y: targets, in synchrony with features (i.e. x[t] and y[t] correspond to the same time)
    """
    # convert to small sequences for training, starting with length 10
    seqs = []; mask_seqs = []; targets = []; mask_ys = []; ys_weights = []; rmse_seqs = []

    # starts at sequence_length and goes until the end
    # for idx in range(min_sequence_length, X.shape[0]+1, 7): # last in range is step
    for idx in range(min_sequence_length, X.shape[0]+1, 1):
        # Sequences
        seqs.append(torch.from_numpy(X[:idx,:]))
        mask_seqs.append(torch.ones(idx))
        # Targets
        y_val = y[idx-min_sequence_length:idx+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER]
        y_ = np.ones((min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER,y_val.shape[1])) * pad_value
        y_[:y_val.shape[0],:] = y_val
        # same for weights
        y_val = y_weights[idx-min_sequence_length:idx+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER]
        ys_weights_ = np.ones((min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER,y_val.shape[1])) * pad_value
        ys_weights_[:y_val.shape[0],:] = y_val
        # save for rmse
        y_val = rmse[idx-min_sequence_length:idx+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER]
        rmse_ = np.ones((min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER,y_val.shape[1])) * pad_value
        rmse_[:y_val.shape[0],:] = y_val
        # mask for y
        mask_y = torch.zeros(min_sequence_length+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER)  # this ensures that 
        mask_y[:len(y_val)] = 1
        targets.append(torch.from_numpy(y_))
        ys_weights.append(torch.from_numpy(ys_weights_))
        rmse_seqs.append(torch.from_numpy(rmse_))
        mask_ys.append(mask_y)

    seqs = pad_sequence(seqs,batch_first=True,padding_value=0).type(dtype)
    mask_seqs = pad_sequence(mask_seqs,batch_first=True,padding_value=0).type(dtype)
    ys = pad_sequence(targets,batch_first=True,padding_value=pad_value).type(dtype)
    mask_ys = pad_sequence(mask_ys,batch_first=True,padding_value=0).type(dtype)
    ys_weights = pad_sequence(ys_weights,batch_first=True,padding_value=pad_value).type(dtype)
    rmse_seqs = pad_sequence(rmse_seqs,batch_first=True,padding_value=pad_value).type(dtype)

    return seqs, mask_seqs, ys, mask_ys, ys_weights, rmse_seqs

# dataset class
class SeqData(torch.utils.data.Dataset):
    def __init__(self, region, X, mask_X, y, mask_y, weight_y, rmse, time_seq):
        self.region = region
        self.X = X
        self.mask_X = mask_X
        self.y = y
        self.mask_y = mask_y
        self.weight_y = weight_y
        self.rmse = rmse
        self.time = time_seq

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.region[idx],
            self.X[idx, :, :],
            self.mask_X[idx],
            self.y[idx],
            self.mask_y[idx],
            self.weight_y[idx],
            self.rmse[idx],
            self.time[idx]
        )

########################################################
#           state/national level data 
########################################################


def load_df(region,ew_start_data,ew_end_data,temporal='daily'):
    """ load and clean data"""
    if temporal=='daily':
        df = pd.read_csv(datapath,low_memory=False)
    elif temporal=='weekly':
        df = pd.read_csv(datapath_weekly,low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    if SMOOTH_MOVING_WINDOW:
        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='full')[:-w+1]
        # smooth
        # df.loc[:,'positiveIncr'] = moving_average(df.loc[:,'positiveIncr'].values,SMOOTH_WINDOW)
        df.loc[:,'death_jhu_incidence'] = moving_average(df.loc[:,'death_jhu_incidence'].values,SMOOTH_WINDOW)
    return df

def get_state_train_data(region,pred_week,ew_start_data=EW_START_DATA,temporal='daily'):
    """ get processed dataframe of data + target as array """
    # import data
    region = str.upper(region)
    pred_week=convert_to_epiweek(pred_week) 
    ew_start_data=convert_to_epiweek(ew_start_data)
    df = load_df(region,ew_start_data,pred_week,temporal)
    # select targets
    # targets = df.loc[:,['positiveIncr','death_jhu_incidence']].values
    targets = df.loc[:,['death_jhu_incidence']].values
    # now subset based on input ew_start_data
    df = df[macro_features]
    return df, targets

def get_state_test_data(region,pred_week,temporal='daily'):
    """
        @ param pred_week: prediction week
    """
    pred_week=convert_to_epiweek(pred_week)
    # import smoothed dataframe
    df = load_df(region,pred_week,pred_week+4,temporal)
    new_cases = df.loc[:,'positiveIncr'].values
    new_deaths = df.loc[:,'death_jhu_incidence'].values
    return new_cases, new_deaths

def get_train_targets_all_regions(pred_week,temporal='daily'):
    deaths_all_regions = {}
    for region in regions:
        _, targets = get_state_train_data(region,pred_week,temporal=temporal)
        deaths_all_regions[region] = targets[:,1]  # index 1 is inc deaths
    return deaths_all_regions

def get_train_features_all_regions(pred_week):
    features_all_regions = {}
    for region in regions:
        df, _ = get_state_train_data(region,pred_week)
        features_all_regions[region] = df.to_numpy()
    return features_all_regions


if __name__ == "__main__":
    pass