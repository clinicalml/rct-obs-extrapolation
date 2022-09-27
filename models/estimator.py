import pandas as pd 
import numpy as np
from scipy.stats import norm
import sys 

from falsifier import Falsifier

sys.path.append('../data/')
from DataModule import DataModule, test_params

# sklearn logistic regression (propensity score)
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.utils import resample
from itertools import product
from models import Model

class CATE:
    
    def __init__(self, data_module, strata_metadata=[], strata=[], params={}):
        '''
            Args: 
                data_module: instance of DataModule; precondition is that 
                user has already run the generation and simulation on 
                instance.
                
                e.g. strata: [
                    (('b.marr','==',1,False),('bw','<',2000,True)), 
                    (('b.marr','==',1,False),('bw','>=',2000,True)),
                    (('b.marr','==',0,False),('bw','<',2000,True)),
                    (('b.marr','==',0,False),('bw','>=',2000,True))            
                ]
                
                # enforce that RCT supported strata go first
                e.g. strata_metadata = [
                    ('lbw, married',True),
                    ('hbw, married',True),
                    ('lbw, single',False),
                    ('hbw, single',False)
                ]
        '''
        self.data = data_module.get_datasets()
        self.strata = []
        for stratum in strata: 
            new_stratum = []
            for elem in stratum: 
                col, op, cutoff, norm_truth = elem
                if norm_truth: 
                    new_elem = (col, op, data_module.get_normalized_cutoff(col,cutoff))
                    new_stratum.append(new_elem)
                else: 
                    new_stratum.append((col,op,cutoff))
            self.strata.append(tuple(new_stratum))
        self.strata_metadata = strata_metadata
        self.params = params
    
    def get_strata_names(self, only_rct=False): 
        if only_rct: 
            return [elem[0] for elem in self.strata_metadata if elem[1]]
        return [elem[0] for elem in self.strata_metadata]
        
    def rct_estimate(self, rct_table, y_name='y_rct', trt_name='treat', full = False): 
        return self._compute_rct_estimates(rct_table, y_name=y_name, trt_name=trt_name, full = full) # full: return all subgroup GATES if True

    def _hp_selection(self,
                      data, 
                      test_size=0.2, 
                      seed=42, 
                      model_name='LogisticRegression', 
                      hp={},
                      model_type='binary'): 
        
        X   = data['X']; y = data['y']        
        tr_idxs, val_idxs = train_test_split(np.arange(X.shape[0]),\
                            test_size=test_size,random_state=seed,stratify=y)
        
        X_train, X_val = X[tr_idxs], X[val_idxs]
        y_train, y_val = y[tr_idxs], y[val_idxs]
        best_hp = None # to store best hps 

        param_names = hp.keys()
        param_lists = [hp[k] for k in param_names]
        print('')
        for elem in product(*param_lists): 
            print(f'[trying hp {elem} for {model_name}]')
            params = {k:elem[i] for i,k in enumerate(param_names)}
            params['input_dim'] = X_train.shape[1]
            
            model = Model(model_name, hp=params, model_type=model_type)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_val)
            metric = model.compute_metric(y_val, y_predict)
            
            if best_hp is None or metric > best_hp[0]: 
                best_hp = (metric, params)

        print(f'best hp: {best_hp[1]}')
        return best_hp[1]
    
    def _get_dataframe(self, final_data, column_names, ihdp = True): 
        # column wise
        if len(final_data[0]) == 4: 
            concat_final = [np.concatenate((elem[0],elem[1],elem[2],elem[3]),axis=1) \
                        for elem in final_data]
            all_cols       = ['y_hat', 'y_obs']+list(column_names[3:])
        elif len(final_data[0]) == 2: 
            concat_final = [np.concatenate((elem[0],elem[1]),axis=1) \
                        for elem in final_data]
            if ihdp:
                all_cols = ['y_hat'] + list(column_names[2:])
            else:
                all_cols = ['y_hat'] + list(column_names[2:])
                all_cols = [f for f in all_cols if \
                     f not in ['ID', 'OS', 'HRTARM', 'EVENT'] and \
                    not f.endswith('_E') and not f.endswith('_DY')]
        else: 
            raise ValueError('final data passed wrong number of things')
        # row wise
        final_data_new = np.concatenate(concat_final, axis=0)
        return pd.DataFrame(final_data_new, columns=all_cols)
    
    def _get_strata_indices(self, table): 
        '''
            A function that takes in the desired stratum, table, and outputs the indices for that stratum
        '''
        all_indices = []
        for stratum in self.strata: 
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
        
        return all_indices
    
    def _compute_rct_estimates(self, table, y_name='y_rct', trt_name='treat', full = False): 
        idxs_outer = self._get_strata_indices(table)
        assert len(idxs_outer) == len(self.strata), \
            'mismatched number of indices and number of subgroups'
        rct_table_subgroups = [table[idxs] for i,idxs in \
                enumerate(idxs_outer) if full or self.strata_metadata[i][1]]
        thetas = []; sds = []
        for subgroup_table in rct_table_subgroups: 
            Y0 = subgroup_table[subgroup_table[trt_name] == 0][y_name].values
            Y1 = subgroup_table[subgroup_table[trt_name] == 1][y_name].values
            theta_sg = np.mean(Y1) - np.mean(Y0)
            sd_sg    = np.sqrt(np.var(Y0) / len(Y0) + np.var(Y1) / len(Y1))
            thetas.append(theta_sg); sds.append(sd_sg)
            
        return thetas, sds
    
    def _compute_obs_estimates(self, obs_table):         
        idxs_outer = self._get_strata_indices(obs_table)
        obs_table_subgroups = [obs_table[idxs] for i,idxs in enumerate(idxs_outer)]
        thetas = []; sds = []
        for subgroup_table in obs_table_subgroups: 
            Yhat_sg = subgroup_table['y_hat'].values
            thetas.append(np.mean(Yhat_sg))
            sds.append(np.std(Yhat_sg)/np.sqrt(Yhat_sg.shape[0]))
        return thetas, sds             
    
    def _get_data(self, obs_table, rct_table=None, pool=False, ihdp = True): 
        ## TODO: clean this up; can use another helper function, for example
        if not pool: 
            N = obs_table['treat'].values.shape[0]
            groups = self._get_strata_indices(obs_table)
            G = np.zeros(obs_table.shape[0])
            for i in range(len(groups)):
                G = G + groups[i].values*i
            X = obs_table.drop(columns=['y_obs','y1_obs','y0_obs','treat'],inplace=False).values
            Y = obs_table['y_obs'].values
            T = obs_table['treat'].values
            return N, G, X, Y, T, None, groups
        
        if ihdp:
            # renaming, dropping columns
            rct_table = rct_table.copy()
            num_covariates = self.params['num_continuous'] + self.params['num_binary']
            rct_table.insert(loc=0, column=f'S', value=np.zeros((rct_table['treat'].shape[0],)))
            obs_table.insert(loc=0, column=f'S', value=np.ones((obs_table['treat'].shape[0],)))
            rct_table.drop(columns=['y1_rct','y0_rct'], inplace=True)
            obs_table.drop(columns=['y1_obs','y0_obs'], inplace=True)
            rct_rename = {'y_rct': 'y_hat'}
            rct_rename.update({f'xprime_rct{i+1}':f'xprime{i+1}' \
                                                for i in range(num_covariates)})
            obs_rename = {'y_obs': 'y_hat'}
            obs_rename.update({f'xprime_obs{i+1}':f'xprime{i+1}' \
                                                for i in range(num_covariates)})
            rct_table.rename(columns=rct_rename, inplace=True)
            obs_table.rename(columns=obs_rename, inplace=True)

        pooled_table = pd.concat((obs_table, rct_table),axis=0,sort=False).reset_index(drop=True)
        pooled_table = pooled_table.dropna(axis=1) # will drop additional columns *_DY as well
        
        # get final values
        if ihdp:
            N = pooled_table['treat'].values.shape[0]
            groups = self._get_strata_indices(pooled_table)
            G = np.zeros(pooled_table.shape[0])
            for i in range(len(groups)):
                G = G + groups[i].values*i
            X = pooled_table.drop(columns=['y_hat','S','treat'],inplace=False).values
            Y = pooled_table['y_hat'].values
            T = pooled_table['treat'].values
            S = pooled_table['S'].values
            return N, G, X, Y, T, S, groups
        else:
            N = pooled_table['HRTARM'].values.shape[0]
            groups = self._get_strata_indices(pooled_table)
            G = np.zeros(pooled_table.shape[0])
            for i in range(len(groups)):
                G = G + groups[i].values*i
            X = pooled_table.drop(columns=['ID', 'OS', 'HRTARM', 'EVENT']\
                 + [f for f in pooled_table.columns.values if f.endswith('_E') or f.endswith('_DY')],\
                    inplace=False).values
            Y = pooled_table['EVENT'].values
            T = pooled_table['HRTARM'].values
            S = pooled_table['OS'].values
            return N, G, X, Y, T, S, groups
    
    def obs_estimate_reweight(self, 
                                obs_table, 
                                rct_table=None, 
                                feat_importance=False): 
        # splitting procedure 
        cvk = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.params['cross_fitting_seed'])
        
        
        if self.params['reweighting']: 
            # TODO: return obs_table from this function
            N, G, X, Y, T, S, groups = self._get_data(obs_table, rct_table, pool=True, ihdp = self.params['ihdp'])
        else: 
            N, G, X, Y, T, _, groups = self._get_data(obs_table, pool=False)

        ## Do DOUBLE MACHINE LEARNING. 
        # use test indices to get estimated outcomes 
        # gather outcomes after loop 
        final_data = []
        '''
        Grct = G[np.where(S==0)]
        Gobs = G[np.where(S==1)]
        Gobs_ov = Gobs[np.where(Gobs <= 1)]
        p1 = np.where(Grct==1)[0].shape[0] / Grct.shape[0]
        p0 = np.where(Grct==0)[0].shape[0] / Grct.shape[0]
        p1_obs = np.where(Gobs_ov==1)[0].shape[0] / Gobs_ov.shape[0]
        p0_obs = np.where(Gobs_ov==0)[0].shape[0] / Gobs_ov.shape[0]
        '''

        for train_idx, test_idx in cvk.split(X, Y): 
            Xtrain, Ytrain, Ttrain, Gtrain, Strain = X[train_idx], Y[train_idx], T[train_idx], G[train_idx], S[train_idx]
            Xtest, Ytest, Ttest, Gtest, Stest = X[test_idx], Y[test_idx], T[test_idx], G[test_idx], S[test_idx]
            
            source_idxs = np.where(Strain == 1); target_idxs = np.where(Strain == 0)
            Xtrain_obs, Ytrain_obs = Xtrain[source_idxs], Ytrain[source_idxs]
            Ttrain_obs, Gtrain_obs = Ttrain[source_idxs], Gtrain[source_idxs]
            source_idxs = np.where(Stest == 1); target_idxs = np.where(Stest == 0)
            Xtest_obs, Ytest_obs = Xtest[source_idxs], Ytest[source_idxs]
            Ttest_obs, Gtest_obs = Ttest[source_idxs], Gtest[source_idxs]
            Xtest_rct, Ytest_rct = Xtest[target_idxs], Ytest[target_idxs]
            Ttest_rct, Gtest_rct = Ttest[target_idxs], Gtest[target_idxs]
            
            # fit response surface model and prop score model on train data 
            ## prop score p(T|X)
            print('\nHP search for propensity model')
            data_prop = {'X': Xtrain_obs, 'y': Ttrain_obs}
            best_hp_prop = self._hp_selection(data_prop, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'], 
                      model_name = self.params['propensity_model']['model_name'],
                    # best hp
                      hp = self.params['propensity_model']['hp'],
                      model_type=self.params['propensity_model']['model_type'])
            g = Model(self.params['propensity_model']['model_name'], hp=best_hp_prop,\
                 model_type=self.params['propensity_model']['model_type'])
            g.fit(Xtrain_obs, Ttrain_obs)
            
            ## selection model and eta; prob(S=1|X)
            overlap_groups = [i for i,x in enumerate(self.strata_metadata) if x[1]]
            overlap_idxs   = np.array([idx for idx,elem in enumerate(Gtrain) \
                                       if elem in overlap_groups])
            Xtrain_ov  = Xtrain[overlap_idxs]
            Strain_ov  = Strain[overlap_idxs]
            Gtrain_ov  = Gtrain[overlap_idxs]

            print('\nHP search for selection model')
            data_sel = {'X': Xtrain_ov, 'y': Strain_ov}
            best_hp_sel = self._hp_selection(data_sel, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'], 
                      model_name = self.params['selection_model']['model_name'],
                    # best hp
                      hp = self.params['selection_model']['hp'],
                      model_type=self.params['selection_model']['model_type'])
            s = Model(self.params['selection_model']['model_name'], hp=best_hp_sel,\
                 model_type = self.params['selection_model']['model_type'])
            s.fit(Xtrain_ov, Strain_ov)

            # Resize Xtrain, Ytrain and Ttrain according to the subgroups
            # only resize S = 1
            resize_index = []
            resize_size = int(np.floor(len(Xtrain_obs)/4))
            np.random.seed(self.params['cross_fitting_seed'])
            for i in range(len(groups)):
                resize_index.extend(np.random.choice(np.where(Gtrain_obs==i)[0],size = resize_size))

            Xtrain_obs = Xtrain_obs[resize_index]
            Ytrain_obs = Ytrain_obs[resize_index]
            Ttrain_obs = Ttrain_obs[resize_index]

            Y1train_obs = Ytrain_obs[Ttrain_obs == 1]
            Y0train_obs = Ytrain_obs[Ttrain_obs == 0]
            X1train_obs = Xtrain_obs[Ttrain_obs == 1, :]
            X0train_obs = Xtrain_obs[Ttrain_obs == 0, :]

            ## response surface model (T-learner)
            data_resp1 = {'X': X1train_obs , 'y': Y1train_obs}
            data_resp0 = {'X': X0train_obs , 'y': Y0train_obs}
            
            if len(self.params['response_surface_1']['hp'].keys()) != 0:  # Obsolete?
                print('\nHP search for response surface model')
                best_hp_resp1 = self._hp_selection(data_resp1, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'],
                      model_name=self.params['response_surface_1']['model_name'], 
                      hp = self.params['response_surface_1']['hp'],
                      model_type=self.params['response_surface_1']['model_type'])
                best_hp_resp1['input_dim'] = X1train_obs.shape[1]
            else: 
                print(f'No hp selection for response surface 1. Fitting with default values.')

            if len(self.params['response_surface_0']['hp'].keys()) != 0:  # Obsolete?
                print('\nHP search for response surface model')
                best_hp_resp0 = self._hp_selection(data_resp0, 
                      test_size=0.2, 
                      seed=self.params['cross_fitting_seed'], 
                      model_name=self.params['response_surface_0']['model_name'], 
                      hp = self.params['response_surface_0']['hp'],
                      model_type=self.params['response_surface_0']['model_type'])
                best_hp_resp0['input_dim'] = X0train_obs.shape[1]
            else: 
                print(f'No hp selection for response surface 0. Fitting with default values.')
            f1  = Model(self.params['response_surface_1']['model_name'], hp=best_hp_resp1, model_type=self.params['response_surface_1']['model_type'])
            f0  = Model(self.params['response_surface_0']['model_name'], hp=best_hp_resp0, model_type=self.params['response_surface_0']['model_type'])
            f1.fit(X1train_obs, Y1train_obs)
            f0.fit(X0train_obs, Y0train_obs)

            if feat_importance: 
                ## look at feature importance of g and f1 
                col_names = obs_table.drop(columns=['ID', 'OS', 'HRTARM', 'EVENT'] \
                 + [f for f in obs_table.columns.values if f.endswith('_E') or f.endswith('_DY')], \
                    inplace=False).columns.values
                prop_feat_idxs = np.argsort(g.model.feature_importances_)[::-1][:200] 
                resp_feats_idxs = np.argsort(f1.model.feature_importances_)[::-1][:200]
                proposed_confounders = np.array(list(set(prop_feat_idxs) & set(resp_feats_idxs)))
                confounder_names = col_names[proposed_confounders]
                print(confounder_names)

                # get balance of confounders
                for confounder in confounder_names: 
                    c = obs_table[confounder].values
                    if len(np.unique(c)) == 2 and (np.sum(c) / c.shape[0]) > 0.3 \
                        and (np.sum(c) / c.shape[0]) < 0.7: 
                        print(f'percentage of 1s: {np.sum(c) / c.shape[0]} for {confounder}')
                    else: 
                        print(f'average value: {np.mean(c)} for {confounder}')

            ######## 
            ## Estimate Y_hat # f -- response suface, g -- prop score, s -- selection model
            overlap_idxs   = np.array([idx for idx,elem in enumerate(Gtest_obs) \
                                       if elem in overlap_groups])
            Xtest_obs_ov = Xtest_obs[overlap_idxs]
            Ytest_obs_ov = Ytest_obs[overlap_idxs]
            Ttest_obs_ov = Ttest_obs[overlap_idxs]
            Gtest_obs_ov = Gtest_obs[overlap_idxs]
            # Xt1_obs_ov = np.concatenate((Xtest_obs_ov,np.ones((Xtest_obs_ov.shape[0],1))),axis=1)
            # Xt0_obs_ov = np.concatenate((Xtest_obs_ov,np.zeros((Xtest_obs_ov.shape[0],1))),axis=1)
            
            # for correction factor
            a1 = 1-np.mean(Strain_ov[np.where(Gtrain_ov == 1.)]) # P(S=0|G=1)
            a0 = 1-np.mean(Strain_ov[np.where(Gtrain_ov == 0.)]) # P(S=0|G=0)
            
            ## ingredients
            Xt_pred_phat = s.predict(Xtest_obs_ov) # p(S=1|X)
            Xt1_pred_ehat_obs = g.predict(Xtest_obs_ov) # p(T=1|X)    
            Xt1_pred_ghat_obs = f1.predict(Xtest_obs_ov); Xt0_pred_ghat_obs = f0.predict(Xtest_obs_ov)
            eta_obs_ov = Gtest_obs_ov / a1 + (1-Gtest_obs_ov) / a0 # Not modular, fails when validation group index not {0,1}
            Y1_hat_obs_ov = eta_obs_ov * ( Ttest_obs_ov*(1-Xt_pred_phat)/(Xt_pred_phat*Xt1_pred_ehat_obs) \
                                       * (Ytest_obs_ov - Xt1_pred_ghat_obs) ) 
            Y0_hat_obs_ov = eta_obs_ov * ( (1-Ttest_obs_ov)*(1-Xt_pred_phat)/(Xt_pred_phat*(1-Xt1_pred_ehat_obs)) \
                                       * (Ytest_obs_ov - Xt0_pred_ghat_obs) ) 
            Yhat_obs_ov = Y1_hat_obs_ov - Y0_hat_obs_ov
            
            # Xt1_rct = np.concatenate((Xtest_rct,np.ones((Xtest_rct.shape[0],1))),axis=1)
            # Xt0_rct = np.concatenate((Xtest_rct,np.zeros((Xtest_rct.shape[0],1))),axis=1)
            Xt1_pred_ghat_rct = f1.predict(Xtest_rct); Xt0_pred_ghat_rct = f0.predict(Xtest_rct)
            eta_rct = Gtest_rct / a1 + (1-Gtest_rct) / a0 # Finicky as well (line 390)
            Y1_hat_rct = eta_rct * (Xt1_pred_ghat_rct)
            Y0_hat_rct = eta_rct * (Xt0_pred_ghat_rct)
            Yhat_rct   = Y1_hat_rct - Y0_hat_rct 
            Yhat_ov = np.concatenate((Yhat_obs_ov,Yhat_rct))
            Xtest_ov = np.concatenate((Xtest_obs_ov,Xtest_rct))
            
            ## Estimate Y_hat for extrapolated groups 
            # Xt1 = np.concatenate((Xtest_obs,np.ones((Xtest_obs.shape[0],1))),axis=1)
            # Xt0 = np.concatenate((Xtest_obs,np.zeros((Xtest_obs.shape[0],1))),axis=1)
            Xt1_pred_f = f1.predict(Xtest_obs); Xt0_pred_f = f0.predict(Xtest_obs)
            Xt_pred_g = g.predict(Xtest_obs)
            rs_signal  = Xt1_pred_f - Xt0_pred_f
            ipw_signal = Ttest_obs*(Ytest_obs - Xt1_pred_f) / Xt_pred_g - \
                    (1-Ttest_obs)*(Ytest_obs - Xt0_pred_f) / (1-Xt_pred_g)
            Yhat_obs = rs_signal + ipw_signal            
            ex_idxs  = np.array([idx for idx,elem in enumerate(Gtest_obs) \
                                       if elem not in overlap_groups])

            if len(ex_idxs) != 0: 
                Yhat_obs_ex = Yhat_obs[ex_idxs]
                Xtest_obs_ex = Xtest_obs[ex_idxs]
                Yhat = np.concatenate((Yhat_ov, Yhat_obs_ex))
                Xtest_concat = np.concatenate((Xtest_ov, Xtest_obs_ex),axis=0)
            else: 
                Yhat = Yhat_ov 
                Xtest_concat = Xtest_ov
            
            # append to list
            final_data.append((Yhat[:,None], Xtest_concat))
            
        print(f'number of tuples: {len(final_data)}')
        print(f'shapes of tuple 1: {[elem.shape for elem in final_data[0]]}')
        final_obs_table = self._get_dataframe(final_data, obs_table.columns.values[1:], ihdp = self.params['ihdp'])
        all_thetas, all_sds = self._compute_obs_estimates(final_obs_table)        
        return all_thetas, all_sds
    
    def obs_estimate(self, obs_table, rct_table=None, cross_fitting_seed=42): 
        # splitting procedure 
        cvk = KFold(n_splits=3, shuffle=True, random_state=cross_fitting_seed)
        N, G, X, Y, T, _, groups = self._get_data(obs_table, pool=False)

        ## Do DOUBLE MACHINE LEARNING. 
        # use test indices to get estimated outcomes 
        # gather outcomes after loop 
        final_data = []
        for train_idx, test_idx in cvk.split(obs_table.values): 
            Xtrain, Ytrain, Ttrain, Gtrain = X[train_idx], Y[train_idx], T[train_idx], G[train_idx]
            Xtest, Ytest, Ttest = X[test_idx], Y[test_idx], T[test_idx]
            
            # Resize Xtrain, Ytrain and Ttrain according to the subgroups
            resize_index = []
            resize_size = int(np.floor(len(Xtrain)/4))
            for i in range(len(groups)):
                resize_index.extend(np.random.choice(np.where(Gtrain==i)[0],size = resize_size))

            Xtrain = Xtrain[resize_index]
            Ytrain = Ytrain[resize_index]
            Ttrain = Ttrain[resize_index]
            
            # fit response surface model and prop score model on train data 
            
            ## prop score p(T|X)
            data_prop = {'X': Xtrain, 'y': Ttrain}
            best_hp_prop = self._hp_selection(data_prop, 
                      test_size=0.2, 
                      seed=cross_fitting_seed, 
                      model_name='LogisticRegression', 
                      hp={'C': [1.,0.1, 0.01, 0.001]},
                      model_type='prop_score')
            g = Model('LogisticRegression', hp=best_hp_prop, model_type='prop_score')
            g.fit(Xtrain, Ttrain)
            
            ## response surface model 
            XTtrain = np.concatenate((Xtrain,Ttrain[:,None]),axis=1)
            
            # response surface model 
            model_name = self.params['response_surface']['model']
            hp_rs      = self.params['response_surface']['hp']
            data_resp = {'X': XTtrain, 'y': Ytrain}
            best_hp_resp = {}
            if len(hp_rs.keys()) != 0:
                best_hp_resp = self._hp_selection(data_resp, 
                      test_size=0.2, 
                      seed=cross_fitting_seed, 
                      model_name=model_name, 
                      hp = hp_rs,
                      model_type='response_surface')
                best_hp_resp['input_dim'] = XTtrain.shape[1]
            else: 
                print(f'No hp selection for {model_name}. Fitting with default values.')
            f  = Model(model_name, hp=best_hp_resp, model_type='response_surface')
            
            ## reweighting model 
            if self.params['reweighting']: 
                pass 
            
            # Estimate Y_hat 
            f.fit(XTtrain, Ytrain)
            Xt1 = np.concatenate((Xtest,np.ones((Xtest.shape[0],1))),axis=1)
            Xt0 = np.concatenate((Xtest,np.zeros((Xtest.shape[0],1))),axis=1)
            Xt1_pred_f = f.predict(Xt1); Xt0_pred_f = f.predict(Xt0)
            Xt_pred_g = g.predict_proba(Xtest)[:,1]
            rs_signal  = Xt1_pred_f - Xt0_pred_f
            ipw_signal = Ttest*(Ytest - Xt1_pred_f) / Xt_pred_g - \
                    (1-Ttest)*(Ytest - Xt0_pred_f) / (1-Xt_pred_g)
            if self.params['reweighting']: 
                pass
            Yhat_k = rs_signal + ipw_signal
            
            # append to list
            final_data.append((Yhat_k[:,None], Ytest[:,None], Ttest[:,None], Xtest))
            
        print(f'number of tuples: {len(final_data)}')
        print(f'shapes of tuple 1: {[elem.shape for elem in final_data[0]]}')
        final_obs_table = self._get_dataframe(final_data, obs_table.columns.values)
        all_thetas, all_sds = self._compute_obs_estimates(final_obs_table)        
        return all_thetas, all_sds
    
    def true_cate(self, table):
        indices = self._get_strata_indices(table)
        table_subgroups = [table[idxs] for i,idxs in enumerate(indices)]
        thetas = []
        for j,subgroup_table in enumerate(table_subgroups): 
            
            in_rct = self.strata_metadata[j][1]
            if not self.params['reweighting'] or in_rct: 
                thetas.append((subgroup_table['y1_rct'].values-subgroup_table['y0_rct'].values).mean())
            else: 
                new_prob = (1 - subgroup_table['sex']*self.params['reweighting_factor'])\
                            *(1 - subgroup_table['cig']*self.params['reweighting_factor'])\
                            *(1 - subgroup_table['work.dur']*self.params['reweighting_factor'])
                new_prob = new_prob / np.sum(new_prob)
                thetas.append(((subgroup_table['y1_rct'].values-subgroup_table['y0_rct'].values)*new_prob).sum())
        return thetas
        
  
'''
Can use these functions as unit tests.

def test_indices_strata_function(): 
    bw_norm = self.strata[0][1][2]
    sg1 = (table['bw']<bw_norm) & (table['b.marr']==1.)
    sg2 = (table['bw']>=bw_norm) & (table['b.marr']==1.)
    sg3 = (table['bw']<bw_norm) & (table['b.marr']==0.)
    sg4 = (table['bw']>=bw_norm) & (table['b.marr']==0.)
    all_indices_true = [sg1,sg2,sg3,sg4]

    for i in range(4): 
        idxs = all_indices[i]; idxs_true = all_indices_true[i]
        for j in range(len(idxs)): 
            assert idxs[j] == idxs_true[j]

'''
            
            
if __name__ == '__main__': 
    
    params = {
        'num_continuous': 4,
        'num_binary': 3,
        'omega': -23,
    #     'gamma_coefs': [1.5,2.5,3.5,4.5,5.5],
        'gamma_coefs': [0.1,0.2,0.5,0.75,1.],
        'gamma_probs': [0.2,0.2,0.2,0.2,0.2], 
        'confounder_seed': 0,
        'beta_seed': 4,
        'noise_seed': 0,
        'obs_dict': {
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
  
#     root = '~/proj/rct_obs_causal/data/'
    # generate 1 simulation of data
    test_params(params)
    ihdp_data = DataModule(params = params)
    ihdp_data.generate_dataset()
    data_dicts = ihdp_data.get_datasets()
    print(f'data generation parameters: {params}')
    # estimate RCT CATE
    strata = [
        (('b.marr','==',1,False),('bw','<',2000,True)), 
        (('b.marr','==',1,False),('bw','>=',2000,True)),
        (('b.marr','==',0,False),('bw','<',2000,True)),
        (('b.marr','==',0,False),('bw','>=',2000,True))            
    ]
    strata_metadata = [
        ('lbw, married',True),
        ('hbw, married',True),
        ('lbw, single',False),
        ('hbw, single',False)
    ]
    cate_estimator = CATE(ihdp_data, strata=strata, \
        strata_metadata=strata_metadata, params=params)
    theta_hats, sd_hats = cate_estimator.rct_estimate(rct_table=data_dicts['rct-partial'])
    strata_names_rct = cate_estimator.get_strata_names(only_rct=True)
    
    # estimate OBS CATE 
    full_theta_obs = []; full_sd_obs = []
    for k,obs_table in enumerate(data_dicts['obs']): 
        if params['reweighting']: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate_reweight(
                                            obs_table=obs_table, \
                                            rct_table=data_dicts['rct-partial'])
        else: 
            thetas_obs, sds_obs = cate_estimator.obs_estimate(obs_table=obs_table)

        full_theta_obs.append(thetas_obs); full_sd_obs.append(sds_obs)
    strata_names_obs = cate_estimator.get_strata_names()
                                       
    print('==============================')
    # "correct CATES"
    print('')
    print('True CATE')
    true_cate_vector = cate_estimator.true_cate(table = data_dicts['rct-full'])
    for j,strata_name in enumerate(strata_names_obs):
        print(f'True CATE of {strata_name} group: {true_cate_vector[j]}')
    print('')
    '''
    obs_tables = data_dicts['obs']
    for i,obs_table in enumerate(obs_tables): 
        print(f'CATE for OBS {i+1}')
        ihdp_data.compute_cate(table=obs_table, data_type='obs', print_true=False)
        print('==============================')
    '''
    
    print('')
    print('RCT CATEs')
    for j,strata_name in enumerate(strata_names_rct): 
        print(f'mean (std) CATE of {strata_name} group: {theta_hats[j]} ({sd_hats[j]})')                           
    print('')
    
    obs_tables = data_dicts['obs']
    for i,obs_table in enumerate(obs_tables): 
        print(f'OBS {i+1} CATEs:')
        for j,strata_name in enumerate(strata_names_obs):
            print(f'mean (std) CATE of {strata_name} group: {full_theta_obs[i][j]} ({full_sd_obs[i][j]})')
        print('=================================')
    
    # run algorithm
    falsifier = Falsifier(alpha=0.05)
    falsifier.run_validation(theta_hats, sd_hats, full_theta_obs, \
                             full_sd_obs, strata_names=strata_names_obs, return_acc = False)
    

    
