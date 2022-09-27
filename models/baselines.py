import pandas as pd 
import numpy as np
from falsifier import Falsifier
from scipy.stats import norm

# Ours - select + conservative (Ours)
# baseline 1 - no select + meta (MetaAnalyzer)
# baseline 2 - no select + no meta (SimpleBaseline)
# baseline 3 - select + meta (EvolvedMetaAnalyzer)

class Baseline:
    
    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        
    def compute_intervals(self, thetas = [], sds = [], strata_names = []): 
        raise NotImplementedError('need to implement compute_intervals function.')

class MetaAnalyzer(Baseline): 
    
    def compute_intervals(self, thetas = [], sds = [], strata_names = []):
        '''
            Output meta-analyzed CATE estimates
        '''
        return self._meta_analysis(thetas, sds, strata_names)
    
    def _meta_analysis(self, thetas, sds, strata_names, baseline_name='Meta-analysis'):
        '''
            TODO: comment, core meta analysis function
        '''
        num_strata = len(strata_names)
        num_obs = len(thetas)
        if num_obs == 0: 
            return ([np.nan for i in range(num_strata)],[np.nan for i in range(num_strata)])
        uci_out = [] ; lci_out = []
        for i in range(num_strata):
            theta_pile = [] ; var_pile = []
            for j in range(num_obs):
                theta_pile.append(thetas[j][i])
                var_pile.append(sds[j][i]**2)
            theta_pile = np.asarray(theta_pile)
            var_pile = np.asarray(var_pile)
            
            if len(theta_pile) != 1: 
                # Calculate heterogeneity variance with Dersimonian-Laird estimator
                theta_hat_fixed = (theta_pile/var_pile).sum()/(1/var_pile).sum()
                Q = ((theta_pile-theta_hat_fixed)**2/var_pile).sum()
                n0 = (Q-len(theta_pile)+1)
                n1 = (1/var_pile).sum()
                n2 = (1/var_pile/var_pile).sum()
                tausq = max(0, n0 / ( n1 - n2 / n1 )  )   
                var_pile = var_pile + tausq

            theta_hat = (theta_pile/var_pile).sum()/(1/var_pile).sum()
            sd_hat = np.sqrt(1/(1/var_pile).sum())
            uci_out.append(theta_hat + norm.ppf(1-self.alpha/2) * sd_hat)
            lci_out.append(theta_hat - norm.ppf(1-self.alpha/2) * sd_hat)
            print(f'{baseline_name} {(1-self.alpha)*100}% confidence interval for {strata_names[i]}: {lci_out[i]}, {uci_out[i]}')

        return (lci_out, uci_out)

class SimpleBaseline(Baseline): 
    # need to use this for width of confidence intervals + cove. prob.
    
    def compute_intervals(self, thetas = [], sds = [], strata_names = []): 
        '''
            Output simple baseline where we report all studies and 
            just return max of upper bound and min of lower bound.
        '''
        num_strata = len(strata_names) 
        num_obs    = len(thetas)
        uci_out = []; lci_out = []
        
        for d in range(num_strata): 
            max_uci_list = []; min_lci_list = []
            for k in range(num_obs): 
                theta_hat = thetas[k][d]
                sd_hat    = sds[k][d]
                max_uci_list.append(theta_hat + norm.ppf(1-self.alpha/2) * sd_hat)
                min_lci_list.append(theta_hat - norm.ppf(1-self.alpha/2) * sd_hat)
            
            uci_out.append(max(max_uci_list))
            lci_out.append(min(min_lci_list))
            print(f'Simple Baseline {(1-self.alpha)*100}% confidence interval for {strata_names[d]}: {lci_out[d]}, {uci_out[d]}')
        
        return (lci_out, uci_out)
    
class EvolvedMetaAnalyzer(MetaAnalyzer): 
    '''
        Selection + meta-analysis
    '''
    def compute_intervals(self, thetas = [], 
                                sds = [], 
                                strata_names = [], 
                                thetas_rct = [],
                                sds_rct = []): 
        # first, do filtering 
        falsifier = Falsifier(alpha=self.alpha)
        test_dim = len(thetas_rct)
        theta_obs_overlap = [np.asarray(elem[:test_dim]) for elem in thetas]
        sd_obs_overlap    = [np.asarray(elem[:test_dim]) for elem in sds]
        accepted_obs = falsifier.falsify(np.asarray(thetas_rct), \
                                         np.asarray(sds_rct), \
                                         theta_obs_overlap, \
                                         sd_obs_overlap)
        thetas_selected = [thetas[i] for i in accepted_obs]
        sds_selected = [sds[i] for i in accepted_obs]
        
        # next, do meta analysis
        return self._meta_analysis(thetas_selected, 
                                    sds_selected, 
                                    strata_names, 
                                    baseline_name='ExOCS')