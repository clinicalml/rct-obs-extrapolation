import pandas as pd 
import numpy as np
import sys 


class DataModule:
    
    def __init__(self, 
                 root = '/afs/csail.mit.edu/u/z/zeshanmh/research/rct_obs_causal/data/',
                 params = {
                        'num_continuous': 4,
                        'num_binary': 3, 
                        'omega': -23, 
                        'gamma_coefs': [0.5,1.5,2.5,3.5,4.5],
                        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
                        'confounder_seed': 0, 
                        'beta_seed': 0, 
                        'noise_seed': 4, 
                        'obs_dict': {
                            'num_obs': 2, 
                            'sizes': [1.,1.], 
                            'confounder_concealment': [0,3], # will be concealed according to ordering of coefficients
                            'missing_bias': [False, False]
                        }, 
                        'rct_dict': { 
                            'size': 1.
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
        
    def _get_confounding_variables(self, probs=[0.5,0.5], binary=True, num_samples=50): 
        Xprime = np.random.choice([0,1],size=num_samples,p=probs)
        if binary: 
            return Xprime
        normal1 = np.random.normal(3,1,size=num_samples)
        normal2 = np.random.normal(0,1,size=num_samples)
        return Xprime*normal1 + (1-Xprime)*normal2
    
    def _generate_confounders(self, data_type='rct', index='0'): 
        base_table = self.ihdp
        probs_trt = [0.5,0.5]; probs_ctr = [0.5,0.5]
        
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
            probs_trt = [0.25,0.75]; probs_ctr = [0.75,0.25]
        elif data_type == 'rct': 
            new_prob = np.ones((self.orig_size,))/self.orig_size
            expand_row = np.random.choice(range(self.orig_size), \
                    size = np.floor(self.orig_size * self.params['rct_dict']['size']).astype('int'),\
                    p = new_prob)
            base_table = self.ihdp.iloc[expand_row,:]

        base_table_trt = base_table[base_table['treat']==1]
        base_table_ctr = base_table[base_table['treat']==0]
        
        # get confounding variable
        Ntrt = base_table_trt['treat'].values.shape[0]
        Nctr = base_table_ctr['treat'].values.shape[0]
        
        Xprime_trts = np.zeros((Ntrt,self.num_continuous+self.num_binary))
        Xprime_controls = np.zeros((Nctr,self.num_continuous+self.num_binary))
        
        # generate continuous confounders - note, diff confounders for each obs dataset (form of concealment)
        t_cont = self._get_confounding_variables(probs=probs_trt, \
                binary=False, num_samples=Ntrt*self.num_continuous)
        c_cont = self._get_confounding_variables(probs=probs_ctr, \
                binary=False, num_samples=Nctr*self.num_continuous)
        Xprime_trts[:,:self.num_continuous] = np.reshape(t_cont, (Ntrt,self.num_continuous))
        Xprime_controls[:,:self.num_continuous] = np.reshape(c_cont, (Nctr,self.num_continuous))
        
        # generate binary confounders
        t_bin = self._get_confounding_variables(probs=probs_trt, \
                    binary=True, num_samples=Ntrt*self.num_binary)
        c_bin = self._get_confounding_variables(probs=probs_ctr, \
                    binary=True, num_samples=Nctr*self.num_binary)
        Xprime_trts[:,self.num_continuous:] = np.reshape(t_bin, (Ntrt,self.num_binary))
        Xprime_controls[:,self.num_continuous:] = np.reshape(c_bin, (Nctr,self.num_binary))
        
        # append 
        Xprime_trts_df = pd.DataFrame(Xprime_trts, columns=[f'xprime_{data_type}{i+1}' \
                            for i in range(self.num_continuous+self.num_binary)])
        Xprime_ctrs_df = pd.DataFrame(Xprime_controls, columns=[f'xprime_{data_type}{i+1}' \
                            for i in range(self.num_continuous+self.num_binary)])
        
        # concat to original data
        ihdp_trt = pd.concat([base_table_trt.reset_index(drop=True), \
                              Xprime_trts_df], axis=1, sort=False)
        
        ihdp_control = pd.concat([base_table_ctr.reset_index(drop=True), \
                              Xprime_ctrs_df], axis=1, sort=False)
        ihdp_table = pd.concat([ihdp_trt, ihdp_control], \
                              ignore_index=True, sort=False)
        #print(ihdp_table)
        # normalization 
        ## TODO: hardcoded for now, may want to change later
        orig_cont = self.ihdp.iloc[:,1:7]
        cont = ihdp_table.iloc[:,1:7]
        ihdp_table.iloc[:,1:7] = (cont - orig_cont.mean())/orig_cont.std() # Normalize continuous variables
        
        return ihdp_table 
    
    def _get_coefs(self): 
        gamma_coefs = self.params['gamma_coefs']
        gamma_probs = self.params['gamma_probs']
        np.random.seed(self.beta_seed)
        # coefs for beta_b and distribution of sampling from Hill
        coefs = np.array([0,0.1,0.2,0.3,0.4])
        probs = np.array([0.6,0.1,0.1,0.1,0.1])
        beta_B = np.random.choice(coefs, size=[self.num_covariates,1], replace=True, p=probs)
        
        # parameters to "tinker with" when generating the data
        gamma = np.random.choice(gamma_coefs, size=[self.num_continuous+self.num_binary,1], \
                                 replace=True, p=gamma_probs)
        
        return beta_B, gamma
        
    def _simulate_outcomes(self,
                         confound_table,
                         beta_B, 
                         gamma,
                         data_type='rct', 
                         response_surface={'ctr': 'non-linear', 'trt': 'linear'}): 
        ## TODO: confounder concealment + missingness bias 
        omega = self.params['omega']
        W  = np.zeros(self.num_covariates)+0.5
        y0 = []; y1 = [] # true
        y  = [] # noise
        
#         np.random.seed(self.noise_seed)
        num_confounders = self.num_continuous+self.num_binary
        for idx, row in confound_table.iterrows(): 
            X = row.values[1:-num_confounders]
            assert X.shape[0] == self.num_covariates

            if response_surface['ctr'] == 'non_linear': 
                mean0 = np.exp(np.matmul((X+W)[None,:],beta_B))[0][0] \
                    + np.matmul(row.loc[f'xprime_{data_type}1':f'xprime_{data_type}{num_confounders}'].values[None,:],gamma)[0][0]
            elif response_surface['ctr'] == 'linear': 
                mean0 = np.matmul((X+W)[None,:],beta_B)[0][0] \
                    + np.matmul(row.loc[f'xprime_{data_type}1':f'xprime_{data_type}{num_confounders}'].values[None,:],gamma)[0][0]
            elif response_surface['ctr'] == 'linear+interaction': 
                row['bw-b.marr-interact'] = row.loc['bw']*row.loc['b.marr']*row.loc['treat']
                row['bw-trt'] = row.loc['bw']*row.loc['treat']
                row['bmarr-trt'] = row.loc['b.marr']*row.loc['treat']
                svars = row.loc[['bw-b.marr-interact','bw-trt', 'bmarr-trt']]
                delta_s = np.array([5.,10.,15.])
                mean0 = np.matmul((X+W)[None,:],beta_B)[0][0] + np.matmul(svars.values[None,:],delta_s[:,None])[0][0] \
                        + np.matmul(row.loc[f'xprime_{data_type}1':f'xprime_{data_type}{num_confounders}'].values[None,:],gamma)[0][0]
                
            if response_surface['trt'] == 'linear': 
                mean1 = np.matmul(X[None,:],beta_B)[0][0] - omega \
                    + np.matmul(row.loc[f'xprime_{data_type}1':f'xprime_{data_type}{num_confounders}'].values[None,:],gamma)[0][0]
            y0.append(mean0); y1.append(mean1)
            y_noise = np.random.normal(row['treat']*mean1 + (1-row['treat'])*mean0, scale=1.)
            y.append(y_noise)

        # inserting potential columns
        confound_table.insert(loc=0, column=f'y0_{data_type}', value=y0)
        confound_table.insert(loc=0, column=f'y1_{data_type}', value=y1)
        confound_table.insert(loc=0, column=f'y_{data_type}', value=y)
        
        return confound_table
    
    def _apply_conf_concealment(self, confound_table, gamma, idx=0): 
        num_remove = self.params['obs_dict']['confounder_concealment'][idx]
        if num_remove == 0: 
            return confound_table 
        idxs_remove = np.argsort(gamma.squeeze())[::-1][:num_remove]
        names_remove = [f'xprime_obs{i+1}' for i in idxs_remove]
        return confound_table.drop(columns=names_remove,inplace=False)
    
    def _split_into_rct_obs(self): 
        ##### DEPRECATED #####
        ## drop columns 
        columns_to_drop = [f'xprime_rct{i+1}' for i in range(self.num_continuous+self.num_binary)]
        # remove y1_rct, y0_rct, xprime_rct
        obs_table = self.ihdp_table.drop(columns=['y_rct','y1_rct','y0_rct']+columns_to_drop,inplace=False) 
        columns_to_drop = [f'xprime_obs{i+1}' for i in range(self.num_continuous+self.num_binary)]
        # remove y1_obs, y0_obs, xprime_obs
        rct_table = self.ihdp_table.drop(columns=['y_obs','y1_obs','y0_obs']+columns_to_drop,inplace=False) 
        print('OBS data')
        print(obs_table)
        print('RCT data')
        print(rct_table)
        
        ## assign the tables
        self.obs_tables = obs_table
        self.rct_table = rct_table 
        
    def generate_dataset(self): 
        # generate data 
        num_datasets  = self.params['obs_dict']['num_obs']+1
        response_surface_dict = self.params['response_surface']
        beta_B, gamma = self._get_coefs()
        
        np.random.seed(self.confounder_seed)
        for k in range(num_datasets): 
            print(f'[Generating confounders for dataset {k+1}.]')
            data_type = 'obs'
            if k == 0: 
                data_type = 'rct'
            confound_table = self._generate_confounders(data_type=data_type, index = k)
            
            print(f'[Simulating outcomes for dataset {k+1}.]')
            confound_table = self._simulate_outcomes(confound_table,
                                     beta_B,
                                     gamma,
                                     data_type=data_type,
                                     response_surface=response_surface_dict)
            if k == 0: 
                confound_table_adjusted = confound_table
            else: 
                confound_table_adjusted = self._apply_conf_concealment(confound_table, gamma, idx=k-1)
            self.final_table_list.append(confound_table_adjusted)
        self.rct_table  = self.final_table_list[0]
        self.obs_tables = self.final_table_list[1:]
        #print(f'[Done!]')
    
    def get_strata_cutoffs(self, root = '/afs/csail.mit.edu/u/z/zeshanmh/research/rct_obs_causal/data/'): 
        ###################
        ### DEPRECATED #### 
        ###################
        ## TODO: determine a more general way of defining strata
        ihdp_table_raw = self._get_original_dataset()
        mean_bw = ihdp_table_raw['bw'].mean()
        std_bw  = ihdp_table_raw['bw'].std()
        bw_norm = (2000 - mean_bw) / std_bw
        strata_cutoffs = {
            'bw_norm': bw_norm
        }
        return strata_cutoffs

    def get_normalized_cutoff(self, col_name, cutoff): 
        mean_ = self.ihdp[col_name].mean()
        std_  = self.ihdp[col_name].std()
        return (cutoff - mean_) / std_
    
    def compute_cate(self, table, data_type='rct', print_true=True, print_emp=True):   
        ## Deprecated
        bw_norm = self.get_normalized_cutoff('bw',2000)
        sg1 = table[(table['bw']<bw_norm) & (table['b.marr']==1.)]
        sg2 = table[(table['bw']>=bw_norm) & (table['b.marr']==1.)]
        sg3 = table[(table['bw']<bw_norm) & (table['b.marr']==0.)]
        sg4 = table[(table['bw']>=bw_norm) & (table['b.marr']==0.)]

        sgs = [sg1,sg2,sg3,sg4]
        sg_names = ['low-bw,married', 'high-bw,married', 'low-bw,single', 'high-bw,single']
        for i,sg in enumerate(sgs): 
            # compute the CATE
            if print_true:
                true_cate = sg[f"y1_{data_type}"].mean() - sg[f"y0_{data_type}"].mean()
                print(f'true CATE for subgroup {i+1} ({sg_names[i]}): {true_cate}')
            if print_emp:
                sg_trt = sg[sg['treat'] == 1]
                sg_ctr = sg[sg['treat'] == 0]
                cate = (sg_trt[f'y_{data_type}'].mean() - sg_ctr[f'y_{data_type}'].mean())
                print(f'noised CATE for subgroup {i+1} ({sg_names[i]}): {cate}')
    
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
        'num_continuous': 4,
        'num_binary': 3, 
        'omega': -23, 
        'gamma_coefs': [0.5,1.5,2.5,3.5,4.5],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'confounder_seed': 0, 
        'beta_seed': 0, 
        'noise_seed': 4, 
        'obs_dict': {
            'num_obs': 1, 
            'sizes': [1.], 
            'confounder_concealment': [0], # will be concealed according to ordering of coefficients
            'missing_bias': [False]
        }, 
        'reweighting': True, 
        'reweighting_factor': 0.25,
        'response_surface': { 
            'ctr': 'linear', 
            'trt': 'linear'
        } 
    }
    
    
    ## subroutine test function for the parameters
    # write assertion statement for 'obs_dict' key 
    # write another assertion statement
    test_params(params)
    
    ihdp_data = DataModule(params=params)
    ihdp_data.generate_dataset()
    data_dicts = ihdp_data.get_datasets()
    print('CATE for RCT')
    ihdp_data.compute_cate(table=data_dicts['rct-full'], data_type='rct')
    print('==============================')
    
    obs_tables = data_dicts['obs']
    for i,obs_table in enumerate(obs_tables): 
        print(f'CATE for OBS {i+1}')
        ihdp_data.compute_cate(table=obs_table, data_type='obs')
        print('==============================')
    
    print(f'data generation parameters: {params}')
    