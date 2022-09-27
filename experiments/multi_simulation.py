import pandas as pd 
import numpy as np
from scipy.stats import norm
import sys 
import os
from itertools import repeat, chain
import simulation_settings as ss
import threading

sys.path.append('../data/')
sys.path.append('../models/')
from falsifier import Falsifier
from baselines import MetaAnalyzer, SimpleBaseline, EvolvedMetaAnalyzer
from estimator import CATE
from DataModule import DataModule, test_params
from multiprocessing import Pool, cpu_count

def one_run(iter_, alpha, save_folder_name, confounder_seed, noise_seed, \
            strata, strata_metadata, params, root): 
    results = []
    print(f'Simulation Number {iter_+1}')
    params['confounder_seed'] = confounder_seed
    params['noise_seed'] = noise_seed

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
        thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)
        full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
    strata_names_obs = cate_estimator.get_strata_names()

    # collecting results 
    true_cates = cate_estimator.true_cate(data_dicts['rct-full'])
    falsifier = Falsifier(alpha=alpha)
    (lci_out_aos, uci_out_aos), (lci_selected, uci_selected), acc = falsifier.run_validation(theta_hats, \
                sd_hats, full_theta_obs, full_sd_obs, strata_names=strata_names_obs, return_acc = True)

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
        results.append({
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
        })

    # Hack to save with parallel processing
    if iter_ % 1 == 0: 
        if 'bias' in ss.save_folder_name: 
            l = sum(params['obs_dict']['confounder_concealment']) / 3
            if l == 4: 
                l = 3
            print(f'saving run number {int(l)}, iteration num: {iter_}')
        elif 'upsize' in ss.save_folder_name: 
            l = params['obs_dict']['sizes'][0]
            print(f'saving for size {l}, iteration num: {iter_}')
        else: 
            l = 0
        R_inter = pd.DataFrame(results)
        print(os.path.join('./simulation_results/'+save_folder_name,f'simulation_multi{l}.csv'))
        R_inter.to_csv(os.path.join('./simulation_results/'+save_folder_name,f'simulation_multi{l}.csv'))
    
    return results

def run_simulation(num_iters=10, alpha = 0.05, root = '', \
                   strata_mod = '', strata_metadata_mod = '', params_mod = '',\
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
    
    params = {
        'num_continuous': 4,
        'num_binary': 3,
        'omega': -23,
        'gamma_coefs': [0.1,0.2,0.5,0.75,1],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'confounder_seed': 0,
        'beta_seed': 4,
        'noise_seed': 0,
        'obs_dict': {
            'num_obs': 5,
            'sizes': [5., 5., 5., 5., 5.],
            'confounder_concealment': [0, 0, 2, 4, 6], # will be concealed according to ordering of coefficients
            'missing_bias': [False, False, False, False, False]
        }, 
        'rct_downsample': 1., 
        'response_surface': {
            'ctr': 'linear', 
            'trt': 'linear',
            'model': 'RandomForestRegressor',
            'hp': {'n_estimators': [200,400], \
                            'min_samples_split': [2,10], \
                            'max_depth': [5,10,20],
                            'max_features': ['auto']}
        }
    }
    print(params_mod)
    if params_mod != '':
        for i in params_mod:
            print(i)
            params[i[0]] = i[1]
    
    confounder_seeds = np.random.choice(range(10000), size=(num_iters,))
    noise_seeds = np.random.choice(range(10000), size=(num_iters,))

    with Pool() as pool: 
        results_all = pool.starmap(one_run, zip(np.arange(num_iters), 
                                          repeat(alpha), 
                                          repeat(save_folder_name),
                                          confounder_seeds,
                                          noise_seeds,
                                          repeat(strata),
                                          repeat(strata_metadata),
                                          repeat(params),
                                          repeat(root)))
    results = list(chain(*results_all)) 
    R = pd.DataFrame(results)
    return R

def parallel_simulation(save_folder_name = '', num_iters = 100, alpha = 0.5, \
                       root = '', strata_mod = '', strata_metadata_mod = '', \
                       params_mod = '', filenames = 'test.csv'):
    
    if not os.path.isdir('simulation_results/'+save_folder_name): 
        os.makedirs('simulation_results/'+save_folder_name)
    R_all = []
    if params_mod == '':
        R_all.append(run_simulation(num_iters, alpha, root, strata_mod, strata_metadata_mod, params_mod, save_folder_name))
    elif len(params_mod) == 1:
        R_all.append(run_simulation(num_iters, alpha, root, strata_mod, strata_metadata_mod, params_mod[0], save_folder_name))
    else:
        for i in range(len(params_mod)): 
            R_all.append(run_simulation(num_iters, alpha, root, strata_mod, \
                         strata_metadata_mod, params_mod[i], save_folder_name))
#         with Pool() as pool: 
#             R_all = pool.starmap(run_simulation, zip(repeat(num_iters), repeat(alpha), \
#                                                  repeat(root), repeat(strata_mod), repeat(strata_metadata_mod), \
#                                                  params_mod, repeat(save_folder_name)))
    if save_folder_name == '':
        return R_all
    for i in range(len(R_all)): 
        R_all[i].to_csv(os.path.join('./simulation_results/'+save_folder_name,f'simulation_multi{i}.csv'))
    

if __name__ == '__main__': 
    parallel_simulation(save_folder_name = ss.save_folder_name,
                          num_iters = ss.num_iters,
                          alpha = ss.alpha,
                          root = ss.root,
                          strata_mod = ss.strata_mod,
                          strata_metadata_mod = ss.strata_metadata_mod,
                          params_mod = ss.params_mod)
