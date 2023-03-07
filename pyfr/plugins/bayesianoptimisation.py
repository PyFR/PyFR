from time import perf_counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root

class BayesianOptimisationPlugin(BasePlugin):
    """ Bayesian Optimisation applied to PyFR.
    """

    name = 'bayesian_optimisation'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix, **data):

        super().__init__(intg, cfgsect, suffix)
        self.comm, self.rank, self.root = get_comm_rank_root()

        cfgostat = 'soln-plugin-optimisation_stats'
        skip_n = self.cfg.getint(cfgostat, 'skip-first-n')     
        last_n = self.cfg.getint(cfgostat, 'capture-last-n')
        
        self.plotter_switch = self.cfg.getbool(cfgsect, 'plotter-switch', False)
        self.optimisables = self.cfg.getliteral(cfgsect, 'optimisables')
        self.bad_sim_multiplier = self.cfg.getfloat(cfgsect, 'bad-sim-multiplier', 2.0)
        self.columns_from_optimisables()

        self.bnds_var = self.cfg.getfloat(cfgsect, 'bounds-variability', 1)
        self._A_lim = self.cfg.getint(cfgsect, 'KG-limit', 2**len(self.optimisables))
        self._B_lim = self.cfg.getint(cfgsect, 'EI-limit', 3**len(self.optimisables))
        self._C_lim = self.cfg.getint(cfgsect, 'PM-limit', 4**len(self.optimisables))

        intg.opt_type = suffix
        if suffix == 'online':
            self.columns.append('tcurr')
            self.index_name = 'tcurr'
            self._toptend = self.cfg.getfloat(cfgostat, 'tend', intg.tend)     
            self._tend = intg.tend     
            self.noptiters_max = ((self._toptend - intg.tcurr)/intg._dt
                                //(skip_n + last_n))
        elif suffix == 'onfline':
            self.columns.append('iteration')
            self.index_name = 'iteration'
            self.noptiters_max = self.cfg.getint(cfgsect, 'iterations', 2*self._C_lim)
            intg.offline_optimisation_complete = False
        elif suffix == 'offline':
            raise NotImplementedError(f'offline not implemented.')
        else:
            raise ValueError('Invalid suffix')

        print(f"KG limit: {self._A_lim}, ", f"EI limit: {self._B_lim}, ",
              f"PM limit: {self._C_lim}", f"stop limit: {self.noptiters_max}",
                )

        if self.rank == self.root:

            import torch
            self.torch = torch

            seed = self.cfg.getint(cfgsect, 'seed', 0)
            self.torch.manual_seed(seed)        # CPU-based generators
            self.torch.random.manual_seed(seed) # GPU-based generators

            # Stress test with gpu too, compare timings
            self.be = self.cfg.get(cfgsect, 'botorch-backend', 'cpu')
            pr = self.cfg.get(cfgsect, 'botorch-precision', 'double')

            if pr == 'double':
                dtyp = torch.float64
            elif pr == 'single':
                dtyp = torch.float32
            else:
                raise ValueError('Invalid dtype')

            self.torch_kwargs = {'dtype': dtyp, 'device': torch.device(self.be)}

            bnd_r = self.cfg.getliteral(cfgsect, 'bounds')
            bnd_init = list(zip(*bnd_r))
            hbnd = list(zip(*self.cfg.getliteral(cfgsect, 'hard-bounds', bnd_r)))
            self._bnds = self.torch.tensor(bnd_init, **self.torch_kwargs)
            self._bbnds = self.torch.tensor(bnd_init, **self.torch_kwargs)
            self._hbnds = self.torch.tensor(hbnd, **self.torch_kwargs)

            self.negw = self.torch.tensor([-1.0], **self.torch_kwargs)
            self.cost_plot = self.cfg.get(cfgsect, 'cost-plot', None)
            self.cumm_plot = self.cfg.get(cfgsect, 'cumm-plot', None)
            self.speedup_plot = self.cfg.get(cfgsect, 'speedup-plot', None)
            self.outf = self.cfg.get(cfgsect, 'history'  , 'bayesopt.csv')
            self.validate = self.cfg.getliteral(cfgsect, 'validate',[])
            self.opt_motive = None # Default first is random
            self.cand_train = True # Default first candidate

        self.deserialise(intg, data)

    def __call__(self, intg):

        if not intg.reset_opt_stats:
        # If optimisation statistics collected by the previous plugin is not reset yet
        # then don't apply Bayesian Optimisation to the data
            return

        self.check_offline_optimisation_status(intg)

        if self.rank == self.root:
            opt_time_start = perf_counter()

            if intg.opt_type == 'online' and intg.bad_sim:
                if np.isnan(intg.opt_cost_mean) and self.df_train.empty:
                    raise ValueError("Initial configuration must be working.")
                tcurr = self.df_train.iloc[-1, self.df_train.columns.get_loc('tcurr')]
            else:
                tcurr = intg.tcurr

            # Convert last iteration data from intg to dataframe
            tested_candidate = self.candidate_from_intg(intg.pseudointegrator)

            t1 =  pd.DataFrame({
                **{f't-{i}': [val] for i, val in enumerate(tested_candidate)},
                't-m': [intg.opt_cost_mean], 
                't-s': [intg.opt_cost_std], 
                'if-train': [self.cand_train], # Training or validation
                },)
            
            if not self.validate == []: 
                self.cand_train = False
                next_candidate = self.validate.pop()
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val

            elif ((len(self.df_train.index)<self._C_lim) or 
                  intg.bad_sim or 
                  loocv_err>0.5 or 
                  kcv_err>0.01):

                # We need to add the latest candidate to the model only if 
                #   it is for training (not validation)
                # Else we add it to the validation set
                if self.cand_train:
                    t_X_Y_Yv = self.process_training_raw(t1)
                    v_X_Y_Yv = self.process_validation_raw()
                else:
                    v_X_Y_Yv = self.process_validation_raw(t1)
                    t_X_Y_Yv = self.process_training_raw()

                # Increase bounds only after exploration
                if len(self.df_train.index)>=self._B_lim:
                    # Set bounds on the basis of training data 
                    self.expand_bounds(self.happening_region(v_X_Y_Yv[0]))

                print(f"len(self.df_train.index)>=self._B_lim: {len(self.df_train.index)>=self._B_lim}")

                self.add_to_model(*t_X_Y_Yv)

                # Get loocv error with the validation set and training-set
                t1['bounds-size'] = self.bounds_size
                t1['LooCV'] = loocv_err = self.loocv_error
                t1['KCV'] = kcv_err = self.kcv_error(*v_X_Y_Yv)
                kcv_err = 0 if kcv_err==None else kcv_err

                # Check if optimisation is performing alright
                if self.df_train.empty:
                    # Initialisation phase
                    self.opt_motive = 'PM'
                    self.cand_train = True
                elif intg.bad_sim:
                    # Fall-back
                    print("Bad simulation.")
                    self.opt_motive = 'PM'
                    self.cand_train = False
                elif len(self.df_train.index)<self._A_lim:
                    # Initialisation phase - I
                    self.opt_motive = 'KG'
                    self.cand_train = True
                elif loocv_err>0.5:
                    # Explorative phase - II
                    if not self.cand_train:
                        self.opt_motive = 'KG'
                        self.cand_train = True
                    else:
                        self.opt_motive = 'PM'
                        self.cand_train = False
                elif len(self.df_train.index)<self._B_lim:
                    # Exploitative phase - I
                    self.opt_motive = 'EI'
                    self.cand_train = True
                elif kcv_err>0.01:
                    # Exploitative phase - II
                    if not self.cand_train:
                        self.opt_motive = 'EI'
                        self.cand_train = True
                    else:
                        self.opt_motive = 'PM'
                        self.cand_train = False
                else:
                    # Finalising phase
                    self.opt_motive = 'PM'
                    self.cand_train = False

                next_candidate, t1['n-m'], t1['n-s'] = self.next_from_model(self.opt_motive)
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                print(f"{self.opt_motive}: {next_candidate}")

                if not self.opt_motive == 'PM':
                    best_candidate, t1['b-m'], t1['b-s'] = self.next_from_model('PM')
                    for i, val in enumerate(best_candidate):
                        t1[f'b-{i}'] = val
                else:
                    # next candidate is the actual best candidate
                    best_candidate = next_candidate
                    t1['b-m'], t1['b-s'] = t1['n-m'], t1['n-s']
                    for i, val in enumerate(best_candidate):
                        t1[f'b-{i}'] = val

            # Finally, use the best tested working candidate in the end
            else:
                next_candidate, t1['n-m'], t1['n-s'] = self.best_tested()
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                
                # If offline optimisation, then abort the simulation at this point. 
                #if intg.opt_type == 'onfline':
                #    intg.abort = True
                    
            t1['opt-time'] = perf_counter() - opt_time_start

            if intg.opt_type == 'online': 
                t1[self.index_name] = tcurr                
            elif intg.opt_type == 'onfline':
                t1[self.index_name] = len(self.df_train.index)+1
            else:
                raise ValueError('Not a valid opt_type')

            t1['cumm-compute-time'] = intg.pseudointegrator._compute_time

            # ------------------------------------------------------------------
            # Add all the data collected into the main dataframe
            self.df_train = pd.concat([self.df_train, t1], ignore_index=True)
            # ------------------------------------------------------------------
            # Post-process dataframe results
            self.df_train['repetition'] = (self.df_train
                                           .groupby(self._t_cols)
                                           .cumcount()+1)

            if self.df_train['LooCV'].count() > self._A_lim:
                # Get a rolling mean of self.df_train['LooCV']
                self.df_train[f'roll{self._A_lim}-diff-LooCV'] = self.df_train['LooCV'].rolling(window=self._A_lim).mean().diff()

                # If last value in self.df_train[f'roll{self._A_lim}-diff-LooCV'] > 0                
                # and if total number of True values is greater than self._A_lim
                if (self.df_train[f'roll{self._A_lim}-diff-LooCV'].iloc[-1] > 0 
                    and self.df_train['if-train'].sum() > self._A_lim  
                    and kcv_err>0.01):

                    # First get the index of the first value in self.df_train['if-train'] that has the value True
                    location = self.df_train['if-train'].index[self.df_train['if-train'] == True][0]

                    # Set the true value to false
                    self.df_train.loc[location, 'if-train'] = False

                    intg._skip_first_n   += 1
                    intg._capture_last_n += 4
                    print(f"Skipping first {intg._skip_first_n} and capturing last {intg._capture_last_n}.")

            # ------------------------------------------------------------------
            # View results as csv file
            self.df_train.to_csv(self.outf, index=False)
            # ------------------------------------------------------------------
            # Notify intg of the latest generated candidate 
            intg.candidate = self._postprocess_ccandidate(list(t1[self._n_cols].values)[0])
            # ------------------------------------------------------------------
            # Finally, plot if required
            if self.plotter_switch:
                if intg.opt_type == 'online' and self.cumm_plot and self.speedup_plot:
                    self.plot_normalised_cummulative_cost()
                    self.plot_overall_speedup(intg)

                if self.cost_plot:
                    self.plot_normalised_cost(intg)
            # ------------------------------------------------------------------

        intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def serialise(self, intg):
        if self.rank == self.root:
            return {'df_train':self.df_train.to_numpy(dtype=np.float64),
                    'bounds':self.bounds,
                    }

    def deserialise(self, intg, data):

        intg.candidate = {}
        self.depth = self.cfg.getint('solver', 'order', 0)

        if self.rank == self.root:

            if bool(data):
                self.df_train = pd.DataFrame(data['df_train'], columns=self.columns)
                self.bounds = data['bounds']

                # Set the candidate to the next candidate if optimisation was being performed in the previous run                
                next_candidate = self.df_train[self._n_cols].tail(1).to_numpy(dtype=np.float64)[0]
                intg.candidate = self._postprocess_ccandidate(next_candidate)
            else:
                self.df_train = pd.DataFrame(columns=self.columns)
            
        if (self.rank == self.root) and len(intg.candidate)>0:
            intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def check_offline_optimisation_status(self, intg):
        if (self.rank == self.root) and (len(self.df_train.index) > self.noptiters_max):
            intg.offline_optimisation_complete = True
        else:
            intg.offline_optimisation_complete = False

        intg.offline_optimisation_complete = self.comm.bcast(
                                           intg.offline_optimisation_complete, 
                                           root = self.root)

    def process_training_raw(self, t1 = None):

        new_df_train = self.df_train[self.df_train['if-train'] == True] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if t1 is not None:
            # Process all optimisables with t1 too
            tX = pd.concat([new_df_train[self._t_cand], t1[self._t_cand]], **args).astype(np.float64).to_numpy()
            tY = pd.concat([new_df_train['t-m'], t1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)
            tYv = pd.concat([new_df_train['t-s'], t1['t-s'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2
        else:
            tX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            tY = new_df_train['t-m'].to_numpy().reshape(-1, 1)
            tYv = new_df_train['t-s'].to_numpy().reshape(-1, 1) ** 2

        if len(tY) > 0:
            # If NaN, then replace with twice the data of worst working candidate
            tY[np.isnan(tY)] = self.bad_sim_multiplier*np.nanmax(tY)
            tYv[np.isnan(tYv)] = self.bad_sim_multiplier*np.nanmax(tYv)
            tYv = tYv.reshape(-1,1)

        return tX, tY, tYv

    def process_validation_raw(self, v1 = None):

        new_df_train = self.df_train[self.df_train['if-train'] == False] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if v1 is not None:
            # Process all optimisables with t1 too
            vX = pd.concat([new_df_train[self._t_cand], v1[self._t_cand]], **args).astype(np.float64).to_numpy()
            vY = pd.concat([new_df_train['t-m'], v1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)
            vYv = pd.concat([new_df_train['t-s'], v1['t-s'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2
        else:
            vX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            vY = new_df_train['t-m'].to_numpy().reshape(-1, 1)
            vYv = new_df_train['t-s'].to_numpy().reshape(-1, 1) ** 2

        if len(vY) > 0:
            # If NaN, then replace with twice the data of worst working candidate
            vY[np.isnan(vY)] = self.bad_sim_multiplier*np.nanmax(vY)
            vYv[np.isnan(vYv)] = self.bad_sim_multiplier*np.nanmax(vYv)
            vYv = vYv.reshape(-1,1)

        return vX, vY, vYv

    def best_tested(self):
        """ 
            Return the best tested candidate and coresponding stats (mean, std)
        """
        grouped_df_train = self.df_train.groupby(self._t_cand)

        candidate = (grouped_df_train.mean(numeric_only = True)['t-m'].idxmin())
        mean = (grouped_df_train.mean(numeric_only = True)['t-m'].min())
        std = (grouped_df_train.std(numeric_only = True).fillna(grouped_df_train.last())
                                                      .at[candidate,'t-s'])

        candidate = [candidate] if isinstance(candidate, float) else list(candidate)
        return candidate, mean, std

    def plot_normalised_cost(self, intg):
        """Plot cost function statistics.
                onfline: wrt number of iterations.
                online: wrt current time. 

            Normalise all of the below plots with the first iteration value.
            Plot the cost functions for the tested candidate.
            Plot the cost [mean, ub, lb] for predicted best candidate.

        """

        base = float(self.df_train.at[0, 't-m'])

        cost_fig, cost_ax = plt.subplots(1,1, figsize = (8,8))

        cost_ax.set_title(f'Online optimisation \n base cost: {str(base)}')
        cost_ax.set_xlabel(self.index_name)
        cost_ax.set_ylabel('Base-normalised cost')

        cost_ax.axhline(y = 1, color = 'grey', linestyle = ':')
        
#            cost_ax.plot(self.df_train[self.index_name], 
#                        self.df_train['b-m']/base, 
#                        linestyle = ':', color = 'green',
#                        label = 'Predicted best mean cost', 
#                        )

#            cost_ax.fill_between(
#                    self.df_train[self.index_name], 
#                    (self.df_train['b-m']-2*self.df_train['b-s'])/base, 
#                    (self.df_train['b-m']+2*self.df_train['b-s'])/base, 
#                    color = 'green', alpha = 0.1, 
#                    label = 'Predicted best candidate''s confidence in cost',
#                        )

        cost_ax.scatter(self.df_train[self.index_name], 
                     self.df_train['t-m']/base, 
                     color = 'blue', marker = '*',
                     label = 'Tested candidate mean cost', 
                    )

        tested_best = (self.df_train['t-m']
                       .expanding().min()
                       .reset_index(drop=True))

        cost_ax.plot(self.df_train[self.index_name], 
                     tested_best/base,
                     linestyle = ':', color = 'blue',
                     label = 'Best tested cost', 
                    )
        # 
        cummin_loc = list(self.df_train['t-m']
                          .expanding().apply(lambda x: x.idxmin()).astype(int))

        tested_best_std = (self.df_train.loc[cummin_loc]['t-s']
                           .reset_index(drop=True))
        cost_ax.fill_between(
                self.df_train[self.index_name], 
                (tested_best-2*tested_best_std)/base, 
                (tested_best+2*tested_best_std)/base, 
                color = 'blue', alpha = 0.04, 
                label = 'Tested best candidate''s confidence in cost',
                    )

        cost_ax.plot(self.df_train[self.index_name], 
                     self.df_train['n-m']/base,
                     linestyle = ':', color = 'green',
                     label = 'Best next cost', 
                    )

        next_index_add = intg._dt if intg.opt_type == 'online' else 1

        cost_ax.fill_between(
                self.df_train[self.index_name] + next_index_add, 
                (self.df_train['n-m']-2*self.df_train['n-s'])/base, 
                (self.df_train['n-m']+2*self.df_train['n-s'])/base, 
                color = 'green', alpha = 0.1, 
                label = 'next candidate''s confidence in cost',
                    )

        self.add_limits_to_plot(cost_ax)

        # Set lower limit for y axis to 0
        cost_ax.set_ylim(bottom = 0)
        #if intg.opt_type == 'onfline':  cost_ax.set_xlim(left = 0)
        #elif intg.opt_type == 'online': cost_ax.set_xlim(left = self._toptstart)

        cost_ax.legend(loc='upper right')
        cost_ax.get_figure().savefig(self.cost_plot)        
        plt.close(cost_fig)
        
    def plot_overall_speedup(self, intg):
        """ 
            If online optimisation is being performed ...
            Plot the optimisation progress in terms of the cummulative cost.
        """

        cumm_fig, cumm_ax = plt.subplots(1, 1, figsize = (8, 8))

        cumm_ax.set_title(f'Online optimisation\nSpeedup wrt base simulation')
        cumm_ax.set_xlabel(self.index_name)
        cumm_ax.set_ylabel('Overall speedup wrt base simulation')

        proj = 1 + self.df_train.index - (self.df_train['t-m']
                                        .isna()
                                        .expanding().sum())

        base = (self.df_train.at[0, 'cumm-compute-time']*proj)

        cumm_ax.axhline(y = 1, color = 'grey', linestyle = ':',)

        ref = float(self.df_train.at[0, 't-m'])

        if (len(self.df_train.index))<self._C_lim:
            m = float(self.df_train[
                self.df_train['t-m'].reset_index(drop=True)
                == self.df_train['t-m'].min()]['t-m'].reset_index(drop=True))

            s = float(self.df_train[
                self.df_train['t-m'].reset_index(drop=True) == 
                self.df_train['t-m'].min()]['t-s'].reset_index(drop=True))

            cumm_ax.axhline(y = ref/m, 
                            linestyle = ':', color = 'green', 
                            )

            cumm_ax.axhspan(ref/(m-2*s), ref/(m+2*s), 
                            color = 'green', alpha = 0.1, 
                            label = 'Possible best tested speed-up'
                            )
        else:        

            m1 = self.df_train.at[self.df_train.last_valid_index(), 'n-m']
            s1 = self.df_train.at[self.df_train.last_valid_index(), 'n-s']

            cumm_ax.axhline(y = ref/m1, 
                            linestyle = ':', color = 'green', 
                            )

            cumm_ax.axhspan(ref/(m1-2*s1), ref/(m1+2*s1), 
                            color = 'green', alpha = 0.1, 
                            label = 'Aiming speed-up',
                            )

        cumm_ax.plot(self.df_train[self.index_name], 
                     base/self.df_train['cumm-compute-time'],
                     linestyle = '-', color = 'g', marker = '*',
                     label = 'compute-time speedup',
                    )

        cumm_ax.plot(self.df_train[self.index_name], 
                     base/(self.df_train['cumm-compute-time'] + 
                           self.df_train['opt-time'].expanding().sum()),
                     linestyle = '-', color = 'b', marker = '*',
                     label = 'overall speedup',
                    )

        self.add_limits_to_plot(cumm_ax)

        cumm_ax.set_ylim(bottom = 0)
        #if intg.opt_type == 'onfline':
        #    cumm_ax.set_xlim(left = 0, right = self.noptiters_max)
        #elif intg.opt_type == 'online':
        #    cumm_ax.set_xlim(left = self._toptstart, right = self._tend)

        cumm_ax.legend(loc='lower right')
        cumm_ax.get_figure().savefig(self.speedup_plot)
        plt.close(cumm_fig)

    def plot_normalised_cummulative_cost(self):
        """ 
            If online optimisation is being performed ...
            Plot the overall speed-up in simulation due to the optimisation

            Compare with base simulation if base simulation data exists
            Else, extrapolate the first iteration data.

        """

        cumm_fig, cumm_ax = plt.subplots(1, 1, figsize = (8, 8))

        cumm_ax.set_title(f'Online optimisation\nCummulative costs')
        cumm_ax.set_xlabel(self.index_name)
        cumm_ax.set_ylabel('cummilative costs (in seconds)')

        proj = self.df_train.index - (self.df_train['t-m']
                                    .isna().expanding().sum() + 1)
        
        cumm_ax.plot(self.df_train[self.index_name], 
                     self.df_train.at[0, 'cumm-compute-time']*proj,
                     linestyle = '--', color = 'grey',
                     label = 'projected compute-time',
                    )

        cumm_ax.plot(self.df_train[self.index_name], 
                     self.df_train['cumm-compute-time'] + 
                     self.df_train['opt-time'].expanding().sum(),
                     linestyle = '-', color = 'blue',
                     label = 'actual cost (compute + optimisation)',
                    )

        cumm_ax.plot(self.df_train[self.index_name], 
                     self.df_train['cumm-compute-time'],
                     linestyle = '-', color = 'green',
                     label = 'compute-cost',
                    )

        cumm_ax.plot(self.df_train[self.index_name], 
                     self.df_train['opt-time'].expanding().sum(),
                     linestyle = '-', color = 'red',
                     label = 'optimisation cost',
                     )

        self.add_limits_to_plot(cumm_ax)

        cumm_ax.set_ylim(bottom = 0)
        #if intg.opt_type == 'onfline':
        #    cumm_ax.set_xlim(left = 0, right = self.noptiters_max)
        #elif intg.opt_type == 'online':
        #    cumm_ax.set_xlim(left = self._toptstart, right = self._tend)

        cumm_ax.legend(loc='lower right')
        cumm_ax.get_figure().savefig(self.cumm_plot)
        plt.close(cumm_fig)

    def add_limits_to_plot(self, ax):
        limits = [(self._A_lim, 'red', 'KG-EI transition'),
                (self._B_lim, 'orange', 'EI-PM transition'),
                (self._C_lim, 'yellow', 'Optimisation turned off')]

        for limit, color, label in limits:
            if len(self.df_train.index) > limit:
                ax.axvline(x=float(self.df_train.at[limit, self.index_name]),
                        color=color, linestyle='-.', label=label)

    def add_to_model(self, tX, tY, tYv):
        """ Fit a Fixed Noise Gaussian Process model on given data

        Args:
            tX (numpy.ndarray): Training input
            tY (numpy.ndarray): Training output
            tYv (numpy.ndarray): Training output variance
        """

        from gpytorch.mlls             import ExactMarginalLogLikelihood
        from botorch.models            import FixedNoiseGP
        from botorch.fit               import fit_gpytorch_model as fit_model
        from botorch.models.transforms import Standardize, Normalize

        self.normalise = Normalize(d=tX.shape[1], bounds=self._bnds)
        self.standardise = Standardize(m=1)

        self.normalise.train(True)
        self.standardise.train(True)
        
        # from botorch.models.transforms import ChainedOutcomeTransform
        # self.trans = ChainedOutcomeTransform(stan=Standardize(m=1))

        self._norm_X = self.normalise.transform(
            self.torch.tensor(tX , **self.torch_kwargs))
        self._stan_Y, self._stan_Yvar = self.standardise.forward(
            self.torch.tensor(tY , **self.torch_kwargs),
            self.torch.tensor(tYv, **self.torch_kwargs))

        self.model = FixedNoiseGP(train_X = self._norm_X, 
                                  train_Y = self._stan_Y, 
                                  train_Yvar = self._stan_Yvar)

        mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, 
                                         model = self.model)

        mll = mll.to(**self.torch_kwargs)                      
        fit_model(mll)

        self.normalise.train(False)
        self.standardise.train(False)

    @property
    def loocv_error(self):
        if (len(self.df_train.index))>=1:
            # Calculate LOOCV error only after the initialising 16 iterations
            # This is to avoid wasting time in the start
            #   when model is definitely not good enough
            return self.cv_folds(self._norm_X, self._stan_Y, self._stan_Yvar
                                 ).detach().cpu().numpy()
        else:
            return None
            
    def kcv_error(self, vX, vY, vYv):
        # Based on processing done like normalise and standardise, process vX

        if len(vY) == 0:
            return None

        _norm_vX = self.normalise.transform(
            self.torch.tensor(vX[-self._A_lim:] , **self.torch_kwargs)) 
        _stan_vY, _stan_vYv = self.standardise.forward(
            self.torch.tensor(vY[-self._A_lim:] , **self.torch_kwargs),
            self.torch.tensor(vYv[-self._A_lim:], **self.torch_kwargs))

        if (len(self.df_train.index))>=1:
            # Calculate LOOCV error only after the initialising 16 iterations
            # This is to avoid wasting time in the start
            #   when model is definitely not good enough
            return self.cv_folds(self._norm_X, self._stan_Y, self._stan_Yvar,
                                 _norm_vX, _stan_vY, _stan_vYv,
                                 ).detach().cpu().numpy()
        else:
            return None
            
    @property
    def bounds(self):
        return self._bnds.detach().cpu().numpy()

    @bounds.setter
    def bounds(self, y):
        self._bnds = self.torch.tensor(y, **self.torch_kwargs)

    def happening_region(self, tX):
        # Create happening region 
        best_cands = tX[-self._A_lim :]

        if len(best_cands) < self._A_lim:
            return

        means, stds = np.mean(best_cands, axis=0), np.std( best_cands, axis=0)
        hr = self.torch.tensor(np.array([means - self.bnds_var * stds,
                                         means + self.bnds_var * stds]), 
                               **self.torch_kwargs)
        
        print('Happening region: ', hr , 'Bounds: ', self._bnds)

        return hr

    def expand_bounds(self, hr):
        if hr is None:
            return

        # Increase bounds to include happening region
        l_bnds_inc_loc = self._bbnds[0, :] > hr[0, :]
        u_bnds_inc_loc = self._bbnds[1, :] < hr[1, :]
        
        self._bnds[0, l_bnds_inc_loc] = hr[0, l_bnds_inc_loc]
        self._bnds[1, u_bnds_inc_loc] = hr[1, u_bnds_inc_loc]

        # Restrict the bounds to hardbounds region
        self._bnds[0, :] = self.torch.max(self._bnds[0, :], self._hbnds[0, :])
        self._bnds[1, :] = self.torch.min(self._bnds[1, :], self._hbnds[1, :])

    @property
    def bounds_size(self):
        return np.prod((self._bnds[1,:]-self._bnds[0,:]).detach().cpu().numpy())
        
    def candidate_from_intg(self, pseudointegrator):
        """ 
            Get candidate used in the last window from the integrator 
            Store the results into a dictionary mapping optimisable name to value
            Use the same optimisable name in modify_configuration plugin
        """
        unprocessed = []

        if any(opt.startswith('cstep:') for opt in self.optimisables):
            n_csteps = sum(opt.startswith('cstep:') for opt in self.optimisables)
            csteps = self._preprocess_csteps(pseudointegrator.csteps, n_csteps)

        for opt in self.optimisables:
            if opt.startswith('cstep:') and opt[6:].isdigit():
                unprocessed.append(csteps[int(opt[6:])])
            elif opt == 'pseudo-dt-max':
                unprocessed.append(pseudointegrator.pintg.Δτᴹ)
            elif opt == 'pseudo-dt-fact':
                unprocessed.append(pseudointegrator.dtauf)
            else:
                raise ValueError(f"Unrecognised optimisable: {opt}")

        return unprocessed

    def _postprocess_ccandidate(self, ccandidate):
        post_processed = {}
        for i, opt in enumerate(self.optimisables):
            if opt.startswith('cstep:') and opt[6:].isdigit():
                post_processed[opt] = ccandidate[i]
            elif opt == 'pseudo-dt-max':
                post_processed[opt] = ccandidate[i]
            elif opt == 'pseudo-dt-fact':
                post_processed[opt] = ccandidate[i]
            else:
                raise ValueError(f"Unrecognised optimisable {opt}")
        return post_processed

    def _preprocess_csteps(self, csteps, n_csteps):

        if n_csteps == 4:
            return csteps[0], csteps[self.depth], csteps[-2], csteps[-1]
        elif n_csteps == 3:
            return csteps[0], csteps[self.depth], csteps[-1]
        elif n_csteps == 2:
            return csteps[self.depth], csteps[-1]
        elif n_csteps == 1:
            return csteps[-1],

    def next_from_model(self, type):

        from botorch.acquisition           import qKnowledgeGradient
        from botorch.acquisition.objective import ScalarizedPosteriorTransform

        from botorch.acquisition           import qNoisyExpectedImprovement
        from botorch.acquisition.objective import ScalarizedPosteriorTransform

        from botorch.acquisition import PosteriorMean

        if type == 'KG':
            samples   = 1024
            restarts  =   10
            fantasies =  128

            _acquisition_function = qKnowledgeGradient(
                self.model, 
                posterior_transform = ScalarizedPosteriorTransform(weights=self.negw),
                num_fantasies       = fantasies,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'EI':
            samples   = 4096 if self.be == 'cuda' else 1024
            restarts  =   20 if self.be == 'cuda' else   10

            _acquisition_function = qNoisyExpectedImprovement(
                self.model, self._norm_X,
                posterior_transform = ScalarizedPosteriorTransform(weights=self.negw),
                prune_baseline = True,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'PM':
            _acquisition_function = PosteriorMean(self.model, maximize = False)

            raw_samples  = 1024 if self.be == 'cpu' else 4096
            num_restarts =   10 if self.be == 'cpu' else   20

            X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
            X_b    = self.normalise.untransform(X_cand)

        else:
            raise ValueError(f'next_type {type} not recognised')

        return self.substitute_in_model(X_b)

    def optimise(self, _acquisition_function, raw_samples=4096, num_restarts=100):
        from botorch.optim.optimize import optimize_acqf
        X_cand, _ = optimize_acqf(acq_function = _acquisition_function, q = 1,
            bounds = self.normalise.transform(self._bnds),
            num_restarts = num_restarts,raw_samples = raw_samples)
        return X_cand

    def substitute_in_model(self, X_sub):
        """ Accept an untransformed candidate into model. 
            Get an untransformed output.
        """

        X_best = self.normalise.transform(X_sub)
        Y_low, Y_upp = self.model.posterior(X_best).mvn.confidence_region()
        Y_avg = self.model.posterior(X_best).mvn.mean

        Y_m = self.standardise.untransform(Y_avg)[0].squeeze().detach().cpu().numpy()
        Y_l = self.standardise.untransform(Y_low)[0].squeeze().detach().cpu().numpy()
        Y_u = self.standardise.untransform(Y_upp)[0].squeeze().detach().cpu().numpy()
        Y_std = (Y_u - Y_l)/4

        XX = X_sub.detach().cpu().squeeze().tolist()

        if isinstance(XX, float):
            return [XX], Y_m, Y_std
        else:
            return XX, Y_m, Y_std
        
    def cv_folds(self, train_X, train_Y, train_Yvar, 
                 test_X=None, test_Y=None, test_Yvar=None):

        from botorch.cross_validation import CVFolds, gen_loo_cv_folds
        from botorch.cross_validation import batch_cross_validation
        from botorch.models import FixedNoiseGP
        from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        if test_X is None or test_Y is None or test_Yvar is None:
            cv_folds = gen_loo_cv_folds(
                train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        else:
            cv_folds = CVFolds(
                train_X=train_X, test_X=test_X, 
                train_Y=train_Y, test_Y=test_Y, 
                train_Yvar=train_Yvar, test_Yvar=test_Yvar)

        # instantiate and fit model
        cv_results = batch_cross_validation(model_cls=FixedNoiseGP,
            mll_cls=ExactMarginalLogLikelihood, cv_folds=cv_folds,)

        posterior = cv_results.posterior
        mean = posterior.mean
        cv_error = ((cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()
        
        return cv_error

    def columns_from_optimisables(self):

        self.columns = [
            *[f't-{i}' for i in range(len(self.optimisables))],'t-m', 't-s', 
            *[f'n-{i}' for i in range(len(self.optimisables))],'n-m', 'n-s', 
            *[f'b-{i}' for i in range(len(self.optimisables))],'b-m', 'b-s',
            'bounds-size', 
            'opt-time', 'cumm-compute-time', 
            'repetition', 'LooCV',  'KCV', 'if-train',
            ] 

        self._t_cols = list(filter(lambda x: x.startswith('t-'), self.columns))
        self._t_cand = list(filter(lambda x: x.startswith('t-') and x[2:].isdigit(), self.columns))
        
        self._b_cols = list(filter(lambda x: x.startswith('b-'), self.columns))
        self._b_cand = list(filter(lambda x: x.startswith('b-') and x[2:].isdigit(), self.columns))
        
        self._n_cols = list(filter(lambda x: x.startswith('n-'), self.columns))
        self._n_cand = list(filter(lambda x: x.startswith('n-') and x[2:].isdigit(), self.columns))
