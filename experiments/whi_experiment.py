from unittest.result import failfast
import pandas as pd 
import numpy as np
import sys 
import os
import argparse
import time
import copy

from itertools import repeat, combinations
from distutils.util import strtobool
from numpy.random import default_rng
from scipy.stats import norm
sys.path.append('../data/')
sys.path.append('../data/whi/')
sys.path.append('../models/')
from falsifier import Falsifier
from baselines import MetaAnalyzer, SimpleBaseline, EvolvedMetaAnalyzer
from estimator import CATE
from DataModule import DataModule, test_params
from DataModuleWHI import DataModuleWHI
sys.path.append('../models_mmr/')
from falsifier_mmr import FalsifierMMR
from estimator_mmr import OutcomeEstimator
from multiprocessing import Pool 
#from util import *

def get_strata_info(covariate_list): 
    cov_name1, cutoff1 = covariate_list[0].split(',')
    cutoff1 = float(cutoff1)
    cov_name2, cutoff2 = None, None 
    if len(covariate_list) > 1: 
        cov_name2, cutoff2 = covariate_list[1].split(',')
        cutoff2 = float(cutoff2)

    if cutoff1 == 1 or cutoff1 == 0: 
        stratum11 = (cov_name1, '==', 1, False)
        stratum10 = (cov_name1, '==', 0, False)
    else: 
        stratum11 = (cov_name1, '>=', cutoff1, True)
        stratum10 = (cov_name1, '<', cutoff1, True)

    if cutoff2 == 1 or cutoff2 == 0: 
        stratum21 = (cov_name2, '==', 1, False)
        stratum20 = (cov_name2, '==', 0, False)
    else: 
        stratum21 = (cov_name2, '>=', cutoff2, True)
        stratum20 = (cov_name2, '<', cutoff2, True)

    if len(covariate_list) > 1: 
        strata_mod = [
            (stratum11, stratum21), 
            (stratum11, stratum20), 
            (stratum10, stratum21), 
            (stratum10, stratum20)
        ]
    else: 
        strata_mod = [(stratum11,), (stratum10,)]

    
    strata_metadata_mod = []
    for i,elem in enumerate(strata_mod): 
        if len(elem) == 2: 
            str1 = f'{elem[0][0]} {elem[0][1]} {str(elem[0][2])}'
            str2 = f'{elem[1][0]} {elem[1][1]} {str(elem[1][2])}'
            s = str1 + ', ' + str2
        else: 
            s = f'{elem[0][0]} {elem[0][1]} {str(elem[0][2])}'

        if i == 0 or i == 1: 
            strata_metadata_mod.append((s, True))
        else: 
            strata_metadata_mod.append((s, False))
        
    return strata_mod, strata_metadata_mod

def get_strata_Ns(strata, table, data_module): 
    '''
        A function that takes in the desired stratum, table, and 
        outputs the indices for that stratum
    '''
    normalized_strata = []
    for stratum in strata: 
        new_stratum = []
        for elem in stratum: 
            col, op, cutoff, norm_truth = elem
            if norm_truth: 
                new_elem = (col, op, data_module.get_normalized_cutoff(col,cutoff))
                new_stratum.append(new_elem)
            else: 
                new_stratum.append((col,op,cutoff))
        normalized_strata.append(tuple(new_stratum))

    all_indices = []
    for stratum in normalized_strata: 
        idxs_truth = pd.Series([True for _ in range(table.shape[0])])
        for elem in stratum: 
            cutoff = float(elem[2])
            if elem[1] == '==': 
                idxs_truth = idxs_truth&((table[elem[0]]-cutoff).abs()<1e-6)
            elif elem[1] == '<=': 
                idxs_truth = idxs_truth&(table[elem[0]] <= cutoff)
            elif elem[1] == '>=': 
                idxs_truth = idxs_truth&(table[elem[0]] >= cutoff)
            elif elem[1] == '<': 
                idxs_truth = idxs_truth&(table[elem[0]] < cutoff)
            elif elem[1] == '>': 
                idxs_truth = idxs_truth&(table[elem[0]] > cutoff)
            else: 
                raise ValueError('invalid operator (choose one of <,>,<=,>=,==)')
        all_indices.append(idxs_truth)
    
    return [np.where(g.values == 1.)[0].shape[0] for g in all_indices]

def run_experiment(alpha = 0.05, 
                    root = '', 
                    strata_mod = '', 
                    strata_metadata_mod = '', 
                    params_mod = '',
                    save_folder_name = '', 
                    exp_name = '',
                    falsification_type = 'GATE', 
                    downsize=False, 
                    downsize_proportion=0.5,
                    split='train', 
                    bootstrap_seed=10): 
    #best hp -- [lr: 0.01, n_estimators: 50, max_depth: 2, min_samples_leaf: 50, min_samples_split: 50, max_features: sqrt]
    #hp search -- [lr: 0.01; 0.001, n_estimators: 50; 25, max_depth: 2, min_samples_leaf: 50; 100, min_samples_split: 50; 100, max_features: sqrt]
    params = {  'ihdp': False,
                'reweighting': True,
                'selection_seed': 42,
                'cross_fitting_seed': 42,
                'grand_seed': 10, 
                'propensity_model': {
                    'model_name': 'GradientBoostingClassifier',
                    'hp': {
                        'learning_rate': [0.01], 'n_estimators': [50],\
                        'max_depth': [2], 'min_samples_leaf': [50], \
                        'min_samples_split': [50], 'max_features': ['sqrt'],
                        'random_state': [42]
                    },
                    'model_type': 'binary'
                },
                'selection_model': {
                    'model_name': 'GradientBoostingClassifier',
                    'hp': {
                        'learning_rate': [0.01], 'n_estimators': [50],\
                        'max_depth': [2], 'min_samples_leaf': [50], \
                        'min_samples_split': [50], 'max_features': ['sqrt'],
                        'random_state': [42]
                    },
                    'model_type': 'binary'
                },
                'response_surface_1': {
                    'model_name': 'GradientBoostingClassifier',
                    'hp': {
                        'learning_rate': [0.01], 'n_estimators': [50],\
                        'max_depth': [2], 'min_samples_leaf': [50], \
                        'min_samples_split': [50], 'max_features': ['sqrt'],
                        'random_state': [42]
                    },
                    'model_type': 'binary'
                },
                'response_surface_0': {
                    'model_name': 'GradientBoostingClassifier',
                    'hp': {
                        'learning_rate': [0.01], 'n_estimators': [50],\
                        'max_depth': [2], 'min_samples_leaf': [50], \
                        'min_samples_split': [50], 'max_features': ['sqrt'],
                        'random_state': [42]
                    },
                    'model_type': 'binary'
                },
                'obs_dict': {
                    ##### Confounder Concealment Configurations
                    ## [None, 'menstrual', 'age+menstrual', 
                    ##### 'age+body+healthcare', 'age+menstrual+body+prior_hrt+healthcare+lab+dep'], 
                    ## [None, None,None, 'age+body+healthcare', 
                    ##### 'age+menstrual+body+prior_hrt+healthcare+lab+dep'],
                    'resample_seed': [1,2,3,4,5],
                    'confounder_concealment': [None, 'menstrual', 'age+menstrual', 'age+menstrual+healthcare', 'menstrual+age+prior_hrt+healthcare+lab+dep'],
                    'selection_bias': [None, 0.1, 0.3, 0.5, 0.7]  #[None, None, None, 0.5, 0.7],  # p, Y = 0, T = 0
                },
                'save_folder_name': save_folder_name, 
                'exp_name': exp_name, 
                'kernel': 'polynomial'
            }
    # data dicts 
    whi_table = DataModuleWHI(params=params)
    whi_table.process_whi(obs_type='selection_bias', 
                          downsize=downsize, 
                          downsize_proportion=downsize_proportion, 
                          split=split,
                          bootstrap_seed=bootstrap_seed)
    data_dicts = whi_table.get_datasets()

    if falsification_type == 'GATE-debug': 
        run_gate(whi_table, data_dicts, strata_mod, strata_metadata_mod, params, alpha)
    elif falsification_type == 'GATE' or falsification_type == 'GATE-E2': 
        
        # read in covariates or covariate pairs depending on experiment configuration
        results = []
        if 'E2' in falsification_type: 
            l = np.loadtxt('r_covariates.csv', delimiter=';', dtype='str')
            covariates = list(combinations(l,2))
        else: 
            covariates = np.loadtxt('p_covariates.csv', delimiter=';', dtype='str')
            covariates = [[x] for x in covariates]
        
        retained_covariates = []
        rct_partial_orig = copy.deepcopy(data_dicts['rct-partial'])
        for i,p in enumerate(covariates):
            print(f'covariate (or covariate pair): {p}')
            strata_mod, strata_metadata_mod = get_strata_info(p)
            
            ## check prevalence of each group for experiment 2
            if 'E2' in falsification_type: 
                Ns = get_strata_Ns(strata_mod, data_dicts['obs'][0], whi_table)
                print(Ns)
                if Ns[0] < 400 or Ns[1] < 400 or Ns[2] < 400 or Ns[3] < 400: 
                    continue
                print(strata_mod)
                print(Ns)
                print('')
                retained_covariates.append(strata_mod)

            t = time.time() 
            cov    = strata_mod[0][0][0]
            sign   = strata_mod[0][0][1] 
            cutoff = strata_mod[0][0][2]
            if cutoff != 1 and cutoff != 0: 
                cutoff = whi_table.get_normalized_cutoff(cov, cutoff)
            
            if sign == '==': 
                data_dicts['rct-partial'] = rct_partial_orig[rct_partial_orig[cov] == cutoff]
            elif sign == '>=': 
                data_dicts['rct-partial'] = rct_partial_orig[rct_partial_orig[cov] >= cutoff]
            elif sign == '<': 
                data_dicts['rct-partial'] = rct_partial_orig[rct_partial_orig[cov] < cutoff]

            results = run_gate(whi_table, data_dicts, strata_mod, strata_metadata_mod, params, alpha, results)
            print(f'time elapsed: {time.time() - t}')
        
        R_inter = pd.DataFrame(results)
        R_inter.to_csv(os.path.join(f"./whi_results/{params['save_folder_name']}/{params['exp_name']}.csv"))

    elif falsification_type == 'MMR-Absolute' or falsification_type == 'MMR-Contrast': 
        run_mmr(whi_table, data_dicts, params, alpha, falsification_type)
    else: 
        raise ValueError(f'Falsification test {falsification_type} not supported.')


def run_gate(whi_table, 
             data_dicts, 
             strata_mod, 
             strata_metadata_mod, 
             params, 
             alpha, 
             results=[]): 

    # adjust strata 
    if strata_mod == '':
        strata = [
            (('PREG_Yes','==',1,False), ('ALCNOW_Yes','==',1,False)), 
            (('PREG_Yes','==',1,False), ('ALCNOW_Yes','==',0,False)),
            (('PREG_Yes','==',0,False), ('ALCNOW_Yes','==',1,False)), 
            (('PREG_Yes','==',0,False), ('ALCNOW_Yes','==',0,False))            
        ]
    else: 
        strata = strata_mod 
    
    if strata_metadata_mod == '':
        strata_metadata = [
            ('pregnant, alcohol',True), # (group name, whether or not strata is supported on RCT)
            ('pregnant, no alcohol',True),
            ('not pregnant, alcohol',False),
            ('not pregnant, no alcohol',False)
        ]
    else: 
        strata_metadata = strata_metadata_mod

    print(f'[Running GATE for strata -- {strata} -- and metadata -- {strata_metadata}]')

    # define CATE estimator 
    cate_estimator = CATE(whi_table, strata=strata, strata_metadata=strata_metadata, params = params)
    theta_hats, sd_hats = cate_estimator.rct_estimate(rct_table=data_dicts['rct-full'], \
                        y_name='EVENT', trt_name='HRTARM', full = True)
    strata_names_rct = cate_estimator.get_strata_names()

    print('')
    print('RCT CATEs')
    for j,strata_name in enumerate(strata_names_rct): 
        print(f'mean (std) CATE of {strata_name} group: {theta_hats[j]} ({sd_hats[j]})')                           
    print('')
    
    full_theta_obs = []; full_sd_obs = []
    for _ , obs_table in enumerate(data_dicts['obs']): 
        if params['reweighting']: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate_reweight(
                                            obs_table=obs_table, \
                                            rct_table=data_dicts['rct-partial'], \
                                            feat_importance=False)
        else: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)

        full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
    strata_names_obs = cate_estimator.get_strata_names()

    falsifier = Falsifier(alpha=alpha)
    (lci_out_aos, uci_out_aos), (lci_selected, uci_selected), acc, (lci_oracle, uci_oracle) = falsifier.run_validation(
                                                                            theta_hats, 
                                                                            sd_hats, 
                                                                            full_theta_obs, 
                                                                            full_sd_obs, 
                                                                            strata_names=strata_names_obs, 
                                                                            return_acc = True, 
                                                                            return_oracle=True)

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

    # results = []
    for d,stratum in enumerate(strata_metadata): 
        name, in_rct = stratum
        if len(lci_out_aos) == 0:
            lci_aos = np.nan; uci_aos = np.nan
        else:
            lci_aos = lci_out_aos[d]; uci_aos = uci_out_aos[d]

        results_add = {
            'strata_name': name,
            'lci_out_rct': lci_out_rct[d], 
            'uci_out_rct': uci_out_rct[d], 
            'lci_out_aos': lci_aos, 
            'uci_out_aos': uci_aos, 
            'lci_out_meta': lci_out_meta[d], 
            'uci_out_meta': uci_out_meta[d],
            'lci_out_evo': lci_out_evo[d], 
            'uci_out_evo': uci_out_evo[d],
            'lci_out_simple': lci_out_simple[d],
            'uci_out_simple': uci_out_simple[d],
            'lci_out_oracle': lci_oracle[d],
            'uci_out_oracle': uci_oracle[d],
            'accept': acc
        }
        for k in range(len(full_theta_obs)): 
            results_add[f'obs_{k}_estimate'] = full_theta_obs[k][d]
            results_add[f'obs_{k}_sd'] = full_sd_obs[k][d]
        results.append(results_add)

    return results 
    # R_inter = pd.DataFrame(results)
    # R_inter.to_csv(os.path.join(f"./whi_results/unbiased_runs_three_quarter/{params['exp_name']}_{strata[0][0][0]}.csv"))

def run_mmr(whi_table, data_dicts, params, alpha, falsification_type): 

    stacked_tables   = []
    mmr_test_signals = []

    for _, obs_table in enumerate(data_dicts['obs']): 
        oe = OutcomeEstimator(rct_table=data_dicts['rct-partial'], 
                                obs_table=obs_table,
                                params=params)
        stacked_tables.append(oe.get_stacked_table())
        U_obs_a1, U_obs_a0 = oe.estimate_signals(S=1)

        '''
            Part 3: estimation of signals for RCT studies
        '''            
        U_rct_a1, U_rct_a0 = oe.estimate_signals(S=0)
        psi1 = U_obs_a1 - U_rct_a1 
        psi0 = U_obs_a0 - U_rct_a0  
        mmr_test_signals.append((psi0,psi1))
        

    '''
        Part 4: write new falsifier that incorporates MMR test 
        
    '''
    results = []

    falsifier = FalsifierMMR(params=params, alpha=alpha, kernel=params['kernel'], \
                falsification_type=falsification_type)
    for k, _ in enumerate(data_dicts['obs']):  
        t = time.time()
        p_val = falsifier.run_test(stacked_tables[k], mmr_test_signals[k], B=100, parallel=False)
        print(f'time elapsed: {time.time() - t}')   
        covariate_names = ['PREG_Yes', 'ALCNOW_Yes']
        covariate_types = ['binary', 'binary']
        f, X_seq, Xmean, covariate_idx = falsifier.visualize_witness_func(stacked_tables[k], 
                mmr_test_signals[k], covariate_names=covariate_names, covariate_types=covariate_types)
        if falsification_type == 'MMR-Absolute': 
            f1, f0 = f
        
        results_add = {
            'num': k
        }
        results_add[f'rct_obs{k}_pval'] = p_val
        if falsification_type == 'MMR-Absolute': 
            results_add[f'rct_obs{k}_f1'] = f1
            results_add[f'rct_obs{k}_f0'] = f0
        elif falsification_type == 'MMR-Contrast': 
            results_add[f'rct_obs{k}_f'] = f
        results_add[f'rct_obs{k}_Xseq'] = X_seq
        results_add[f'rct_obs{k}_Xmean'] = Xmean
        results_add[f'rct_obs{k}_cov_name'] = covariate_names

        results.append(results_add)
    
    R_inter = pd.DataFrame(results)
    R_inter.to_csv(os.path.join(f"./whi_results/{params['save_folder_name']}/{params['exp_name']}.csv"))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Get falsification test experiment params.')
    parser.add_argument('-f', '--falsification_type', type=str, default='GATE',
                        help='GATE, MMR-Absolute, or MMR-Contrast')
    parser.add_argument('-a', '--alpha', type=float, default=0.05)
    parser.add_argument('-b', '--bootstrap_seed', type=int)
    parser.add_argument('-d', '--downsize', type=strtobool, default=True)
    parser.add_argument('-p', '--downsize_proportion', type=float, default=0.5)
    parser.add_argument('-t', '--split', type=str, default='train')
    parser.add_argument('-s', '--save_folder_name', type=str, default='test')
    parser.add_argument('-e', '--exp_name', type=str, default='test')
    args = parser.parse_args()
    print(f'config - [falsification type: {args.falsification_type}, seed: {args.bootstrap_seed}, exp_name: {args.exp_name}]')
    run_experiment(falsification_type=args.falsification_type, alpha=args.alpha, \
        downsize=args.downsize, downsize_proportion =args.downsize_proportion, split=args.split, \
        bootstrap_seed=args.bootstrap_seed, save_folder_name=args.save_folder_name, exp_name=args.exp_name) 
