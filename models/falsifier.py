import pandas as pd 
import numpy as np
from scipy.stats import norm


class Falsifier(): 
    
    def __init__(self, alpha=0.05): 
        self.alpha = alpha
    
    def run_validation(self, 
                        theta_rct, 
                        sd_rct, 
                        full_theta_obs, 
                        full_sd_obs, 
                        strata_names = [], 
                        return_acc = False, 
                        return_oracle=False): 
        '''
            Precondition: assume overlapping subgroups are in the first d elements
            in each theta_obs vector
        '''
        self.return_acc = return_acc
        self.return_oracle = return_oracle
        assert len(full_theta_obs) != 0, 'need to provide at least one estimated CATE vector'
        for theta_hat in full_theta_obs: 
            assert len(strata_names) == len(theta_hat)
            
        test_dim = len(theta_rct)
        theta_obs_overlap = [np.asarray(elem[:test_dim]) for elem in full_theta_obs]
        sd_obs_overlap    = [np.asarray(elem[:test_dim]) for elem in full_sd_obs]
        accepted_obs = self.falsify(np.asarray(theta_rct), np.asarray(sd_rct), theta_obs_overlap, sd_obs_overlap)
        return self._output_intervals(accepted_obs, full_theta_obs, full_sd_obs, strata_names=strata_names)      
    
    def falsify(self, theta_rct, sd_rct, theta_obs, sd_obs):
        '''
            Run Z-test to determine accepted OBS studies.
            Args: 
                TODO - comment later
        '''
        
        test_dim = len(theta_rct)
        obs_dim = len(theta_obs)
        
        # check dimensions of obs vectors for each obs study
        assert test_dim == len(sd_rct)
        for obs_study in theta_obs: 
            assert test_dim == len(obs_study)
        accepted_obs = []
        for i in range(obs_dim):
            Z = (theta_rct - theta_obs[i]) / np.sqrt(sd_rct ** 2 + sd_obs[i] **2)
            if np.prod(np.abs(Z) < norm.ppf(1 - (self.alpha/4)/test_dim)): # bonferroni correction
                accepted_obs.append(i)
        
        return accepted_obs
    
    def _output_intervals(self, accepted_obs, thetas = [], sds = [], strata_names = []): 
        '''
            Output confidence bounds for specified subgroup.
            Args: 
                thetas - a list of np.arrays representing the estimated 
                            CATE vector for each observational study 
                sds - a list of np.arrays representing the standard 
                            deviation of the estimate above 
        '''
        
        num_strata = len(strata_names)
        uci_selected = [[] for _ in range(num_strata)]
        lci_selected = [[] for _ in range(num_strata)]
        
        uci_out = []; lci_out = []
        if len(accepted_obs) == 0:
            print('All estimates from observational studies are falsified!')
        else:
            print(f'Selected obs studies: {accepted_obs}')
            for i in range(num_strata):
                for j in accepted_obs:
                    uci_selected[i].append(thetas[j][i] + norm.ppf(1-self.alpha/4) * sds[j][i])
                    lci_selected[i].append(thetas[j][i] - norm.ppf(1-self.alpha/4) * sds[j][i])
                uci_out.append(np.max(uci_selected[i]))
                lci_out.append(np.min(lci_selected[i]))
                print(f'Output {(1-self.alpha)*100}% confidence interval for {strata_names[i]}: {lci_out[i]}, {uci_out[i]}')
        ## assumption: unbiased OBS study is the first one; this is oracle 
        print(f'Oracle method intervals:')
        uci_oracle = []; lci_oracle = []
        for i in range(num_strata): 
            uci_oracle.append(thetas[0][i] + norm.ppf(1-self.alpha/4) * sds[0][i])
            lci_oracle.append(thetas[0][i] - norm.ppf(1-self.alpha/4) * sds[0][i])
            print(f'Output {(1-self.alpha)*100}% confidence interval for {strata_names[i]}: {lci_oracle[i]}, {uci_oracle[i]}')

        if self.return_acc and self.return_oracle:
            return (lci_out, uci_out), (lci_selected, uci_selected), \
                ','.join([str(elem) for elem in accepted_obs]), (lci_oracle, uci_oracle)
        elif self.return_oracle: 
            return (lci_out, uci_out), (lci_selected, uci_selected), (lci_oracle, uci_oracle)
        elif self.return_acc: 
            return (lci_out, uci_out), (lci_selected, uci_selected), \
                ','.join([str(elem) for elem in accepted_obs])
    
        return (lci_out, uci_out), (lci_selected, uci_selected)
    