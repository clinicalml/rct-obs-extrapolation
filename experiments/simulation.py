import pandas as pd 
import numpy as np
import sys 
import os
import argparse

from itertools import repeat
from numpy.random import default_rng
from scipy.stats import norm
sys.path.append('../data/')
sys.path.append('../models/')
from falsifier import Falsifier
from baselines import MetaAnalyzer, SimpleBaseline, EvolvedMetaAnalyzer
from estimator import CATE
from DataModule import DataModule, test_params
from multiprocessing import Pool 
from util import *

def run_simulation(num_iters=10, 
                    alpha = 0.05, 
                    root = '', 
                    strata_mod = '', 
                    strata_metadata_mod = '', 
                    params_mod = '',
                    save_folder_name = ''):   
    
    if strata_mod == '':
        strata = [
            (('b.marr','==',1,False),('bw','<',2000,True)), 
            (('b.marr','==',1,False),('bw','>=',2000,True)),
            (('b.marr','==',0,False),('bw','<',2000,True)),
            (('b.marr','==',0,False),('bw','>=',2000,True))            
        ]
    
    if strata_metadata_mod == '':
        strata_metadata = [
            ('lbw, married',True), # (group name, whether or not strata is supported on RCT)
            ('hbw, married',True),
            ('lbw, single',False),
            ('hbw, single',False)
        ]
    
    # params = {
    #     'num_continuous': 4,
    #     'num_binary': 3,
    #     'omega': -23, # [0.2,0.5,.75,1.,1.25]
    #     'gamma_coefs': [0.1,0.2,0.5,0.75,1.],
    #     'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
    #     'grand_seed': 10, # was 10 originally
    #     'confounder_seed': 0,
    #     'beta_seed': 4,
    #     'noise_seed': 0,
    #     'obs_dict': {
    #         'num_obs': 5,
    #         'sizes': [5., 5., 5., 5., 5.],
    #         'confounder_concealment': [0, 0, 2, 4, 6], # will be concealed according to ordering of coefficients
    #         'missing_bias': [False, False, False, False, False]
    #     }, 
    #     'reweighting': True,
    #     'reweighting_factor': 0.2,
    #     'response_surface': {
    #         'ctr': 'linear', 
    #         'trt': 'linear',
    #         'model': 'RandomForestRegressor',
    #         'hp': {'n_estimators': [200,400], \
    #                         'min_samples_split': [2], #,10\ 
    #                         'max_depth': [20], # 5,10
    #                         'max_features': ['auto']}
    #     }
    # }
    params = {
        'num_continuous': 4,
        'num_binary': 3,
        'omega': -23, # [0.2,0.5,.75,1.,1.25]
        # 'gamma_coefs': [0.2,0.5,1.,1.25,1.75],
        'gamma_coefs': [0.2,0.5,1.25,1.75,2.],
        # 'gamma_coefs': [0.1,0.2,0.5,0.75,1.],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'grand_seed': 10, # was 10 originally
        'confounder_seed': 0,
        'beta_seed': 4,
        'noise_seed': 0,
        'obs_dict': {
            'num_obs': 5,
            'sizes': [5.,5.,5.,5.,5.],
            'confounder_concealment': [0,0,3,4,6], # will be concealed according to ordering of coefficients
            'missing_bias': [False, False, False, False, False]
        }, 
        'reweighting': True,
        'reweighting_factor': 0.2,
        'response_surface': {
            'ctr': 'linear', 
            'trt': 'linear',
            'model': 'LinearRegression',
            'hp': {}
        }
    }
    print(params_mod)
    if params_mod != '':
        for i in params_mod:
            print(i)
            params[i[0]] = i[1]
    
    
    rng = default_rng(params['grand_seed'])
    confounder_seeds = rng.choice(range(1000), size=(num_iters,))
    noise_seeds = rng.choice(range(1000), size=(num_iters,))
    
    results = []
    for iter_ in range(num_iters): 
        print(f'Simulation Number {iter_+1}')
        params['confounder_seed'] = confounder_seeds[iter_]
        params['noise_seed'] = noise_seeds[iter_]
        
        # generate dataset
        test_params(params)
        if root != '':
            ihdp_data = DataModule(params = params, root = root)
        else: 
            ihdp_data = DataModule(params = params) # change root path to data and add it as argument
        ihdp_data.generate_dataset()
        data_dicts = ihdp_data.get_datasets()
        print(f'data generation parameters: {params}')
        
        # estimation 
        cate_estimator = CATE(ihdp_data, strata=strata, strata_metadata=strata_metadata, params=params)
        theta_hats, sd_hats = cate_estimator.rct_estimate(rct_table=data_dicts['rct-partial'])
        strata_names_rct = cate_estimator.get_strata_names(only_rct=True)
        
        # estimate OBS CATE 
        full_theta_obs = []; full_sd_obs = []
        for k,obs_table in enumerate(data_dicts['obs']): 
            if params['reweighting']: 
                thetas_obs, sds_obs = cate_estimator.obs_estimate_reweight(obs_table=obs_table, \
                                                                           rct_table=data_dicts['rct-partial'])
            else: 
                thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)
            full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
        strata_names_obs = cate_estimator.get_strata_names()
        
        # collecting results 
        true_cates = cate_estimator.true_cate(data_dicts['rct-full'])
        falsifier = Falsifier(alpha=alpha)
        (lci_out_aos, uci_out_aos), (lci_selected, uci_selected), acc = falsifier.run_validation(
                                                                            theta_hats, 
                                                                            sd_hats, 
                                                                            full_theta_obs, 
                                                                            full_sd_obs, 
                                                                            strata_names=strata_names_obs, 
                                                                            return_acc = True)
        ## baselines
        meta_baseline   = MetaAnalyzer(alpha=alpha)
        simple_baseline = SimpleBaseline(alpha=alpha)
        evo_baseline    = EvolvedMetaAnalyzer(alpha=alpha)
        lci_out_meta, uci_out_meta = meta_baseline.compute_intervals(full_theta_obs, \
                                            full_sd_obs, strata_names=strata_names_obs)
        lci_out_simple, uci_out_simple = simple_baseline.compute_intervals(full_theta_obs, \
                                            full_sd_obs, strata_names=strata_names_obs)
        lci_out_evo, uci_out_evo = evo_baseline.compute_intervals(full_theta_obs, \
                                            full_sd_obs, strata_names_obs, theta_hats, sd_hats)
        lci_out_rct = []; uci_out_rct = []
        for i in range(len(theta_hats)): 
            uci_out_rct.append(theta_hats[i] + norm.ppf(1-alpha/2) * sd_hats[i])
            lci_out_rct.append(theta_hats[i] - norm.ppf(1-alpha/2) * sd_hats[i])

        for d,stratum in enumerate(strata_metadata): 
            name, in_rct = stratum
            if in_rct:
                lci_rct = lci_out_rct[d]; uci_rct = uci_out_rct[d]
            else: 
                lci_rct = np.nan; uci_rct = np.nan
            if len(lci_out_aos) == 0:
                lci_aos = np.nan; uci_aos = np.nan
            else:
                lci_aos = lci_out_aos[d]; uci_aos = uci_out_aos[d]

            results_add = {
                'sim_num': iter_, 
                'strata_num': d+1,
                'strata_name': name,
                'cate_true': true_cates[d], 
                'lci_out_rct': lci_rct, 
                'uci_out_rct': uci_rct, 
                'lci_out_aos': lci_aos, 
                'uci_out_aos': uci_aos, 
                'lci_out_meta': lci_out_meta[d], 
                'uci_out_meta': uci_out_meta[d],
                'lci_out_evo': lci_out_evo[d], 
                'uci_out_evo': uci_out_evo[d],
                'lci_out_simple': lci_out_simple[d],
                'uci_out_simple': uci_out_simple[d],
                'accept': acc
            }
            for k in range(len(full_theta_obs)): 
                results_add[f'obs_{k}_estimate'] = full_theta_obs[k][d]
                results_add[f'obs_{k}_sd'] = full_sd_obs[k][d]
            results.append(results_add)
        
        # Hack to save with parallel processing
        if (iter_+1) % 10 == 0: 
            if 'bias' in save_folder_name: 
                l = sum(params['obs_dict']['confounder_concealment']) / 3
                if l == 4: 
                    l = 3
                print(f'saving run number {int(l)}, iteration num: {iter_+1}')
            elif 'upsize' in save_folder_name: 
                d = params['obs_dict']['sizes'][0]
                dict_ = {0.5:0,1.0:1,3.0:2,5.0:3}
                l = dict_[d]
                print(f'saving for size {d}, iteration num: {iter_+1}')
            else: 
                l = 0
            R_inter = pd.DataFrame(results)
            print(os.path.join('./simulation_results/'+save_folder_name,f'simulation{int(l)}_iter{iter_+1}.csv'))
            R_inter.to_csv(os.path.join('./simulation_results/'+save_folder_name,f'simulation{int(l)}_iter{iter_+1}.csv'))
    R = pd.DataFrame(results)
    return R

def parallel_simulation(save_folder_name = '', 
                        num_iters = 100, 
                        alpha = 0.5, 
                        root = '', 
                        strata_mod = '', 
                        strata_metadata_mod = '', 
                        params_mod = ''):
    
    if not os.path.isdir('simulation_results/'+save_folder_name): 
        os.makedirs('simulation_results/'+save_folder_name)
    R_all = []
    if params_mod == '':
        R_all.append(run_simulation(num_iters, alpha, root, strata_mod, strata_metadata_mod, params_mod, save_folder_name))
    elif len(params_mod) == 1:
        R_all.append(run_simulation(num_iters, alpha, root, strata_mod, strata_metadata_mod, params_mod[0], save_folder_name))
    else:
        R_all = []
        with Pool() as pool: 
            R_all = pool.starmap(run_simulation, zip(repeat(num_iters), repeat(alpha), \
                                                 repeat(root), repeat(strata_mod), repeat(strata_metadata_mod), \
                                                 params_mod, repeat(save_folder_name)))
    if save_folder_name == '':
        return R_all
    for i in range(len(R_all)): 
        R_all[i].to_csv(os.path.join('./simulation_results/'+save_folder_name,f'simulation{i}.csv'))
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demo.yaml',
                        help='give path to simulation config.')
    args   = parser.parse_args()    
    config_info = read_yaml(path=args.config)
    parallel_simulation(save_folder_name = config_info['save_folder_name'],
                          num_iters = config_info['num_iters'],
                          alpha = config_info['alpha'],
                          root = config_info['root'],
                          strata_mod = config_info['strata_mod'],
                          strata_metadata_mod = config_info['strata_metadata_mod'],
                          params_mod = config_info['params_mod'])
