import pandas as pd 
import numpy as np
import sys 

class DataModule:
    '''
        DataModule V2 -- this class implements a different data setup than 
        V1, where we add unobserved confounding synthetically. (see xi parameter)
    '''
    def __init__(self, 
                 root = '/afs/csail.mit.edu/u/z/zeshanmh/research/rct_obs_causal/data/',
                 params = {
                        'num_continuous': 4, 
                        'num_binary': 3,
                        'covariate_unobserved_effect': ['bw'],
                        'omega': -23, 
                        'gamma_coefs': [0.5,1.5,2.5,3.5,4.5],
                        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
                        'confounder_seed': 0, 
                        'beta_seed': 0, 
                        'noise_seed': 4, 
                        'obs_dict': {
                            'num_obs': 2, 
                            'sizes': [1.,1.], 
                            'confounder_concealment': [0,3], # will be concealed according to ordering of coefficients # deprecated
                            'xi': [0.,0.2],
                            'missing_bias': [False, False]
                        }, 
                        'rct_downsample': 1.
                    }): 
        # core init
        self.params = params
        self.root = root
        ihdp_table = pd.read_csv(self.root + 'ihdp.csv')
        self.num_covariates = ihdp_table.columns.values.shape[0]-1
        assert self.num_covariates == 28
        num_obs_datasets = params['obs_dict']['num_obs']
        
        # table init
        self.ihdp             = ihdp_table
        # normalize base table
        orig_cont = self.ihdp.iloc[:,1:7]
        self.ihdp.iloc[:,1:7] = (orig_cont - orig_cont.mean())/orig_cont.std() # Normalize continuous variables
        self.cont_covariate_means  = orig_cont.mean()
        self.cont_covariate_stds   = orig_cont.std()
        self.orig_size        = ihdp_table['treat'].values.shape[0]
        self.final_table_list = []
        self.obs_tables       = None # list of dataframes 
        self.rct_table        = None
        self.rct_table_partial = None
        
        # variable init
        self.num_continuous  = params['num_continuous']
        self.num_binary      = params['num_binary']
        self.confounder_seed = params['confounder_seed']
        self.beta_seed  = params['beta_seed']
        self.noise_seed = params['noise_seed']
    
    def get_means_stds(self): 
        '''
            Get mean and std of continuous covariates.
        '''
        return self.cont_covariate_means, self.cont_covariate_stds
        
    def _get_unobserved_confounders(self, X, num_samples=50): 
        noise = np.random.normal(0,1,size=num_samples)
        theta = np.random.uniform(0,2,size=X.shape[1]) # does this need to be hardcoded 
        Z = np.matmul(X**2, theta[:,None])[:,0] + noise # TODO: maybe come up with a different funcitonal form? 
        return Z

    def _generate_confounders(self, data_type='rct', gamma_prop=[], index=0): 
        base_table = self.ihdp
        
        if data_type == 'obs':
            # TODO: decide if this is what we want to keep or if we don't want to resample when size=1.
            if self.params['reweighting']:     
                new_prob = (1 - self.ihdp['sex']*self.params['reweighting_factor'])\
                        *(1 - self.ihdp['cig']*self.params['reweighting_factor'])\
                        *(1 - self.ihdp['work.dur']*self.params['reweighting_factor'])
                new_prob = new_prob / np.sum(new_prob)
            else: 
                new_prob = np.ones((self.orig_size,))/self.orig_size
            
            expand_row = np.random.choice(range(self.orig_size), \
                    size = np.floor(self.orig_size * self.params['obs_dict']['sizes'][index-1]).astype('int'),\
                    p = new_prob)
            base_table = self.ihdp.iloc[expand_row,:]

        N = base_table.shape[0]
        X_unobserved_effect = base_table[self.params['covariate_unobserved_effect']].values
        Z = self._get_unobserved_confounders(X_unobserved_effect, num_samples=N)
        num_confounders = self.num_continuous+self.num_binary
        
        if data_type == 'obs': 
            # get propensities and regenerate treatments 
            X = base_table.values[:,1:]
            assert len(gamma_prop) == X.shape[1]
            expit_val = np.matmul(X,gamma_prop) \
                        + (Z*self.params['obs_dict']['xi'][index-1])[:,None]
            expit_val = (expit_val - np.mean(expit_val)) / np.std(expit_val)
            pA1_X_Z = 1 / (1 + np.exp(-expit_val))

            new_treat = []
            for i in range(N):
                t = np.random.choice([0,1],p=[1-pA1_X_Z[i].squeeze(),pA1_X_Z[i].squeeze()])
                new_treat.append(t)
            new_treat = np.array(new_treat)
            base_table['treat'] = new_treat
        Z_df = pd.DataFrame(Z, columns=[f'Z_{data_type}{i+1}' for i in range(num_confounders)])
        ihdp_table = pd.concat([base_table.reset_index(drop=True), Z_df], axis=1, sort=False)        

        return ihdp_table 
    
    def _get_coefs_V2(self): 
        gamma_coefs = self.params['gamma_coefs']
        gamma_probs = self.params['gamma_probs']
        np.random.seed(self.beta_seed)

        # gamma for Z (unobserved confounding)
        gamma1 = np.random.choice(gamma_coefs, size=[self.num_continuous+self.num_binary,1], \
                                 replace=True, p=gamma_probs)
        gamma0 = np.random.choice(gamma_coefs, size=[self.num_continuous+self.num_binary,1], \
                                 replace=True, p=gamma_probs) 
        
        # gamma for propensity score
        gamma_prop = np.random.choice(gamma_coefs, size=[self.num_covariates,1], \
                                 replace=True, p=gamma_probs) 
        gamma_prop = gamma_prop 

        # coefs for beta_b and distribution of sampling from Hill
        coefs = np.array([0,0.1,0.2,0.3,0.4])
        probs = np.array([0.6,0.1,0.1,0.1,0.1])
        beta_B = np.random.choice(coefs, size=[self.num_covariates,1], replace=True, p=probs)
        zeta   = np.random.choice(coefs, size=[self.num_covariates,1], replace=True, p=probs)

        return beta_B, zeta, gamma1, gamma0, gamma_prop

    def _simulate_outcomes(self,
                         confound_table,
                         beta_B,
                         zeta,
                         gamma1, 
                         gamma2,
                         data_type='rct', 
                         response_surface={'ctr': 'non-linear', 'trt': 'linear'}): 
        # TODO: response surfaces will need to change (add gamma, for example)
        y0 = []; y1 = [] # true
        y  = [] # noise
        # np.random.seed(self.noise_seed)
        num_confounders = self.num_continuous+self.num_binary
        for idx, row in confound_table.iterrows(): 
            X = row.values[1:-num_confounders]
            assert X.shape[0] == self.num_covariates
            Z = row.values[-1]

            if response_surface['ctr'] == 'linear_unobserved': 
                mean0 = np.matmul(X[None,:],zeta)[0][0] + Z*gamma2[0][0]
            else: 
                raise ValueError('invalid response surface for control given.')

            if response_surface['trt'] == 'linear_unobserved': 
                mean1 = np.exp(np.matmul(X[None,:],beta_B) + 0.5)[0][0] + Z*gamma1[0][0]
            else: 
                raise ValueError('invalid response surface for treatment given.')
            y0.append(mean0); y1.append(mean1)
            y_noise = np.random.normal(row['treat']*mean1 + (1-row['treat'])*mean0, scale=1.)
            y.append(y_noise)

        # inserting potential columns
        confound_table.insert(loc=0, column=f'y0_{data_type}', value=y0)
        confound_table.insert(loc=0, column=f'y1_{data_type}', value=y1)
        confound_table.insert(loc=0, column=f'y_{data_type}', value=y)
        
        return confound_table

    def _apply_conf_concealment(self, confound_table, data_type='obs'): 
        num_remove = self.num_continuous+self.num_binary # number of unobserved confounders 
        assert num_remove >= 1 
        names_remove = [f'Z_{data_type}{i+1}' for i in range(num_remove)]
        return confound_table.drop(columns=names_remove,inplace=False)

    def generate_dataset(self): 
        # generate data 
        num_datasets  = self.params['obs_dict']['num_obs']+1
        response_surface_dict = self.params['response_surface']
        beta_B, zeta, gamma1, gamma2, gamma_prop = self._get_coefs_V2()
        
        np.random.seed(self.confounder_seed)
        for k in range(num_datasets): 
            print(f'[Generating confounders for dataset {k+1}.]')
            data_type = 'obs'
            if k == 0: 
                data_type = 'rct'
            confound_table = self._generate_confounders(data_type=data_type, \
                            gamma_prop=gamma_prop, index = k)
            
            print(f'[Simulating outcomes for dataset {k+1}.]')
            confound_table = self._simulate_outcomes(confound_table,
                                     beta_B,
                                     zeta,
                                     gamma1,
                                     gamma2, 
                                     data_type=data_type,
                                     response_surface=response_surface_dict)
            confound_table_adjusted = self._apply_conf_concealment(confound_table, 
                                        data_type=data_type)
            self.final_table_list.append(confound_table_adjusted)
        self.rct_table  = self.final_table_list[0]
        self.obs_tables = self.final_table_list[1:]
        #print(f'[Done!]')
    
    def get_normalized_cutoff(self, col_name, cutoff): 
        mean_ = self.ihdp[col_name].mean()
        std_  = self.ihdp[col_name].std()
        return (cutoff - mean_) / std_

    def get_datasets(self): 
        # when we return the RCT to user, restrict to only married people 
        self.rct_table_partial = self.rct_table[self.rct_table['b.marr'] == 1.].reset_index(drop=True)
        return {
            'rct-partial': self.rct_table_partial,
            'rct-full': self.rct_table, 
            'obs': self.obs_tables
        }

def test_params(params): 
    obs_dict = params['obs_dict']
    assert len(obs_dict['sizes']) == obs_dict['num_obs'], \
        'Number of specified sizes does not match number of requested obs studies.'
    assert len(obs_dict['confounder_concealment']) == obs_dict['num_obs'], \
        'Number of concealed confounders does not match number of requested obs studies.'
    assert len(obs_dict['missing_bias']) == obs_dict['num_obs'], \
        'Number of missing bias entries not match number of requested obs studies.'
    for n in obs_dict['confounder_concealment']: 
        if n > (params['num_continuous']+params['num_binary']): 
            raise ValueError('invalid confounder concealment value')
    
if __name__ == '__main__': 
    
    params = {
        'num_continuous': 1,
        'num_binary': 0, 
        'covariate_unobserved_effect': ['bw'],
        'omega': -23, # deprecated 
        'gamma_coefs': [0.5,1.5,2.5,3.5,4.5],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'confounder_seed': 0, 
        'beta_seed': 0, 
        'noise_seed': 4, 
        'obs_dict': {
            'num_obs': 2, 
            'sizes': [1.,1.], 
            'confounder_concealment': [0,1], # deprecated, will be concealed according to ordering of coefficients
            'xi': [0.,0.2],
            'missing_bias': [False,False]
        }, 
        'reweighting': True, 
        'reweighting_factor': 0.25,
        'response_surface': { 
            'ctr': 'linear_unobserved', 
            'trt': 'linear_unobserved'
        } 
    }
    
    ## subroutine test function for the parameters
    # write assertion statement for 'obs_dict' key 
    # write another assertion statement
    test_params(params)
    
    ihdp_data = DataModule(params=params)
    ihdp_data.generate_dataset()
    data_dicts = ihdp_data.get_datasets()
    obs_tables = data_dicts['obs']
    print(data_dicts['rct-partial'])
    print(obs_tables[0])
    print(obs_tables[1])

    print(f'data generation parameters: {params}')
     