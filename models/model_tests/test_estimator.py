import pytest 
import sys
import os
sys.path.append('../../data/')
sys.path.append('../')
from DataModule import DataModule 
from estimator import CATE

@pytest.fixture
def params_config(): 
    return {
        'num_continuous': 4,
        'num_binary': 3,
        'omega': -23,
        'gamma_coefs': [0.1,0.2,0.5,0.75,1.],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'confounder_seed': 0,
        'beta_seed': 4,
        'noise_seed': 0,
        'obs_dict': {
            'num_obs': 1,
            'sizes': [1.],
            'confounder_concealment': [0], # will be concealed according to ordering of coefficients
            'missing_bias': [False]
        }, 
        'reweighting': True, 
        'reweighting_factor': .25,
        'response_surface': {
            'ctr': 'non_linear', 
            'trt': 'linear',
            'model': 'MLP',
            'hp': {'hidden_layer_sizes': [(50,50)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [.0001],
                    'learning_rate': ['adaptive'],
                    'learning_rate_init': [1e-3],
                    'max_iter': [600]}
            }
    }

@pytest.fixture
def strata(): 
    return [
        (('b.marr','==',1,False),('bw','<',2000,True)), 
        (('b.marr','==',1,False),('bw','>=',2000,True)),
        (('b.marr','==',0,False),('bw','<',2000,True)),
        (('b.marr','==',0,False),('bw','>=',2000,True))            
    ]

@pytest.fixture
def strata_metadata(): 
    return [
        ('lbw, married',True),
        ('hbw, married',True),
        ('lbw, single',False),
        ('hbw, single',False)
    ]

@pytest.fixture
def ihdp_data_items(params_config): 
    ihdp_data = DataModule(params = params_config)
    ihdp_data.generate_dataset()
    data_dicts = ihdp_data.get_datasets()
    return ihdp_data, data_dicts

@pytest.mark.fast
def test_true_cates(params_config, strata, strata_metadata, ihdp_data_items):
    # compute true gates with the estimator
    ihdp_data, data_dicts = ihdp_data_items  
    cate_estimator = CATE(ihdp_data, strata=strata, \
        strata_metadata=strata_metadata, params=params_config)
    true_cate_vector = cate_estimator.true_cate(table = data_dicts['rct-full'])
    
    # define ground truth gates
    true_cates = {
        'lbw, married': 4.1909, 
        'hbw, married': 1.3253, 
        'lbw, single': 1.2847, 
        'hbw, single': -1.9343
    }

    # assertions
    strata_names_obs = cate_estimator.get_strata_names()
    for j,strata_name in enumerate(strata_names_obs):
        assert true_cate_vector[j] == pytest.approx(true_cates[strata_name], 1e-4)

@pytest.mark.fast
def test_rct_estimates(params_config, strata, strata_metadata, ihdp_data_items): 
    # get rct estimates
    ihdp_data, data_dicts = ihdp_data_items  
    cate_estimator = CATE(ihdp_data, strata=strata, \
        strata_metadata=strata_metadata, params=params_config)
    theta_hats, sd_hats = cate_estimator.rct_estimate(
        rct_table=data_dicts['rct-partial']
    )

    # define ground truth 
    true_thetas = [4.6176, 2.6794]
    true_stds = [0.6751, 0.9975]
    
    # assert that they are approx equal
    strata_names_rct = cate_estimator.get_strata_names(only_rct=True)
    for j,strata_name in enumerate(strata_names_rct): 
        assert theta_hats[j] == pytest.approx(true_thetas[j], 1e-4)
        assert sd_hats[j] == pytest.approx(true_stds[j], 1e-4)

def test_obs_estimates(params_config, strata, strata_metadata, ihdp_data_items): 
    # compute OBS estimates
    ihdp_data, data_dicts = ihdp_data_items  
    cate_estimator = CATE(ihdp_data, strata=strata, \
        strata_metadata=strata_metadata, params=params_config)
    full_theta_obs = []; full_sd_obs = []
    for k,obs_table in enumerate(data_dicts['obs']): 
        if params_config['reweighting']: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate_reweight(
                                            obs_table=obs_table, \
                                            rct_table=data_dicts['rct-partial'])
        else: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)

        full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
    strata_names_obs = cate_estimator.get_strata_names()

    # ground truth 
    true_thetas = [4.0772, 3.1967, -0.0824, -1.3423]
    true_sds    = [1.3138, 1.0116, 1.2364, 0.8306]    

    # assertions    
    obs_tables = data_dicts['obs']
    for i,obs_table in enumerate(obs_tables): 
        for j,strata_name in enumerate(strata_names_obs):
            assert full_theta_obs[i][j] == pytest.approx(true_thetas[j], 1e-3)
            assert full_sd_obs[i][j] == pytest.approx(true_sds[j], 1e-3)