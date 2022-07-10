
import pandas as pd
import numpy as np
import pdb
import os
import math
from data_utils import get_state_test_data
import pandas as pd
import pdb
pd.set_option('precision', 2)

WEEKS_AHEAD = 4 # forecasting horizon in weeks

def parse_results(
        model_name: str,
        disease: str,
        pred_weeks: list, # of strings
        regions: list, # of strings, see format in state_data_processing.py
        suffix: str='',
        target_name: str='cases',
        region_level: str='county',
        ):   
    """
        given a model and a list of weeks and a list of regions, 
        converts to weekly predictions to follow CDC evaluation guidelines
        saves partial results and compute final metrics

        predictions for each week have to be save as csv files
        as preds_{model_name}_{pred_week}.csv'
    """
    model_suffix = '-'.join([model_name,suffix])
    if not os.path.exists(f'./results/{disease}/'):
        os.makedirs(f'./results/{disease}/')
    f1 = open(f'./results/{disease}/results_{model_suffix}.csv', 'w+') 
    f1.write('region,model,pred_week,pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,target1,target2,target3,target4,target5,target6,target7,target8,error1,error2,error3,error4,error5,error6,error7,error8,pearson')
    for region in regions:
        print(f'reading region {region}')
        for pred_week in pred_weeks:
            path = f'./results/{disease}/{region}/'
            file_name = f'preds_{model_name}_{pred_week}_{suffix}.csv'
            df = pd.read_csv(path+file_name,index_col=False,nrows=56)
            if disease=='COVID':
                assert region_level=='state'
                _, new_deaths = get_state_test_data(region,pred_week)
                predictions = df.loc[:,'deaths'].values
                target = new_deaths
                # convert to weekly
                pred1, target1 = np.sum(predictions[:7]), np.sum(target[:7])
                pred2, target2 = np.sum(predictions[7:7*2]), np.sum(target[7:7*2])
                pred3, target3 = np.sum(predictions[7*2:7*3]), np.sum(target[7*2:7*3])
                pred4, target4 = np.sum(predictions[7*3:7*4]), np.sum(target[7*3:7*4])
                pred5, target5 = np.sum(predictions[7*4:7*5]), np.sum(target[7*4:7*5])
                pred6, target6 = np.sum(predictions[7*5:7*6]), np.sum(target[7*5:7*6])
                pred7, target7 = np.sum(predictions[7*6:7*7]), np.sum(target[7*6:7*7])
                pred8, target8 = np.sum(predictions[7*7:7*8]), np.sum(target[7*7:7*8])
            
            error1 = target1 - pred1
            error2 = target2 - pred2
            error3 = target3 - pred3
            error4 = target4 - pred4
            error5 = target5 - pred5
            error6 = target6 - pred6
            error7 = target7 - pred7
            error8 = target8 - pred8

            preds = np.array([pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8])
            targets = np.array([target1,target2,target3,target4,target5,target6,target7,target8])  
            pearson_cor = np.corrcoef(preds,targets)[0,1]

            f1.write(
                '\n{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(
                    region,model_suffix,pred_week,
                    pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,
                    target1,target2,target3,target4,target5,target6,target7,target8,
                    error1,error2,error3,error4,error5,error6,error7,error8,
                    pearson_cor
                )
            )
    f1.close()

    df_all = pd.read_csv(f'./results/{disease}/results_{model_suffix}.csv',header=0)
    df_all['pred_week'] = df_all['pred_week'].astype(str)

    ############################################################################
    ###################   aggregate by model and region  ########################
    ############################################################################

    error = {}

    for key, df in df_all.groupby(['region','model']):
        # normalized rmse as per Naren
        short_nrmse1 = np.sqrt(df[['error1','error2','error3','error4']].pow(2).mean().mean()).item() / (df[['target1','target2','target3','target4']].abs().values+1).mean() # mean().mean().item()
        long_nrmse1 = np.sqrt(df[['error5','error6','error7','error8']].pow(2).mean().mean()).item() / (df[['target5','target6','target7','target8']].abs().values+1).mean() #.mean().item()
        
        # normalized rmse as per https://arxiv.org/pdf/2111.07779.pdf
        short_nrmse2 = np.sqrt(df[['error1','error2','error3','error4']].pow(2).mean().mean()).item() / (df[['target1','target2','target3','target4']].values.max() - df[['target1','target2','target3','target4']].values.min() + 1)
        long_nrmse2 = np.sqrt(df[['error5','error6','error7','error8']].pow(2).mean().mean()).item() / (df[['target5','target6','target7','target8']].values.max() - df[['target5','target6','target7','target8']].values.min() + 1)

        # normal deviation as per Naren
        short_nd = ( (df[['error1','error2','error3','error4']].abs().values).sum() / (df[['target1','target2','target3','target4']].abs().values+1).sum() ).mean()
        long_nd = ( (df[['error5','error6','error7','error8']].abs().values).sum() / (df[['target5','target6','target7','target8']].abs().values+1).sum() ).mean()

        short_mape = (df[['error1','error2','error3','error4']].abs().values / (df[['target1','target2','target3','target4']].abs().values+1)).mean(0).mean()
        long_mape = (df[['error5','error6','error7','error8']].abs().values / (df[['target5','target6','target7','target8']].abs().values+1)).mean(0).mean()

        pearson_cor_median = df[['pearson']].median().values.item()

        error[key] = [short_nrmse1,long_nrmse1,short_nrmse2,long_nrmse2,short_nd,long_nd,short_mape,long_mape,pearson_cor_median]

    cols = ['short_nrmse1','long_nrmse1','short_nrmse2','long_nrmse2','short_nd','long_nd','short_mape','long_mape','pearson']
    df = pd.DataFrame.from_dict(error, orient='index', columns=cols)
    # get back region and model columns
    indexes = list(df.index)
    regions = []; models = []
    for i in range(len(indexes)):
        regions.append(indexes[i][0])
        models.append(indexes[i][1])
    df['region'] = regions
    df['model'] = models
    df = df[['model','region'] + cols] 
    df.reset_index(drop=True, inplace=True)
    df.to_csv('./results/{}/error_table_{}.csv'.format(disease,model_suffix),index=False,float_format='%.2f')   
    df_by_region = df

    ############################################################################
    ########################   aggregate by model only  ############################
    ############################################################################
    error = {}
    for key, df in df_all.groupby(['model']):
        # normalized rmse as per Naren
        short_nrmse1 = np.sqrt(df[['error1','error2','error3','error4']].pow(2).mean().mean()).item() / (df[['target1','target2','target3','target4']].abs().values+1).mean() # mean().mean().item()
        long_nrmse1 = np.sqrt(df[['error5','error6','error7','error8']].pow(2).mean().mean()).item() / (df[['target5','target6','target7','target8']].abs().values+1).mean() #.mean().item()
        
        # normalized rmse as per https://arxiv.org/pdf/2111.07779.pdf
        short_nrmse2 = np.sqrt(df[['error1','error2','error3','error4']].pow(2).mean().mean()).item() / (df[['target1','target2','target3','target4']].values.max() - df[['target1','target2','target3','target4']].values.min() + 1)
        long_nrmse2 = np.sqrt(df[['error5','error6','error7','error8']].pow(2).mean().mean()).item() / (df[['target5','target6','target7','target8']].values.max() - df[['target5','target6','target7','target8']].values.min() + 1)

        # normal deviation as per Naren
        short_nd = ( (df[['error1','error2','error3','error4']].abs().values).sum() / (df[['target1','target2','target3','target4']].abs().values+1).sum() ).mean()
        long_nd = ( (df[['error5','error6','error7','error8']].abs().values).sum() / (df[['target5','target6','target7','target8']].abs().values+1).sum() ).mean()

        short_mape = (df[['error1','error2','error3','error4']].abs().values / (df[['target1','target2','target3','target4']].abs().values+1)).mean(0).mean()
        long_mape = (df[['error5','error6','error7','error8']].abs().values / (df[['target5','target6','target7','target8']].abs().values+1)).mean(0).mean()

        pearson_cor_median = df[['pearson']].median().values.item()

        error[key] = [short_nrmse1,long_nrmse1,short_nrmse2,long_nrmse2,short_nd,long_nd,short_mape,long_mape,pearson_cor_median]

    cols = ['short_nrmse1','long_nrmse1','short_nrmse2','long_nrmse2','short_nd','long_nd','short_mape','long_mape','pearson']
    df = pd.DataFrame.from_dict(error, orient='index', columns=cols)
    df = df.reset_index()
    df.to_csv('./results/{}/error_table_overall_{}.csv'.format(disease,model_suffix),index=False,float_format='%.2f')   
    df_overall = df

    return df_by_region, df_overall 


def eval_models(disease,target_name,region_level,models,regions,weeks):

    results = []; overall_results = []
    for model, suffix in models.items():
        by_region, overall,  = parse_results(
            model_name=model,
            disease=disease,
            pred_weeks=weeks,
            regions=regions,
            suffix=suffix,
            region_level=region_level,
            target_name=target_name
        )
        results.append(by_region)
        overall_results.append(overall)
    df_overall = pd.concat(overall_results) # 

    ''' aggregation over all regions and weeks'''
    print('\ndf_overall')
    print(df_overall)
    df_overall.to_csv(f'./results/{disease}/overall.csv',float_format='%.2f')

if __name__ == "__main__":

    # =====================================================
    # =============== COVID =================

    # =====================================================
    
    disease = 'COVID'
    target_name = 'deaths'
    region_level = 'state'

    models = {
        'EINN':'exp1',  # model and experiment number
    }

    regions = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']
    for x in ['DC','AK','WY','MT']:
        regions.remove(x)

    weeks = ['202036','202038','202040','202042','202044','202046','202048','202050','202052','202101','202103','202105','202107','202109']

    eval_models(disease,target_name,region_level,models,regions,weeks)

   

    
    