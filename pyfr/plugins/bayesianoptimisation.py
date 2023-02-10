from time import perf_counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root

class BayesianOptimisationPlugin(BasePlugin):
    name = 'bayesian_optimisation'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix):

        super().__init__(intg, cfgsect, suffix)

        cfgostat = 'soln-plugin-optimisation_stats'
        skip_n = self.cfg.getint(cfgostat, 'skip-first-n', 10)     
        last_n = self.cfg.getint(cfgostat, 'capture-last-n', 40)

        self.optimisables = self.cfg.getliteral(cfgsect, 'optimisables')
        self.bnds_var = self.cfg.getfloat(cfgsect, 'bounds-variability', 2)

        self.comm, self.rank, self.root = get_comm_rank_root()

        # Read the initial candidate and the validation candidate
        #   This is a list of tuple of 4 floats
        self.init_cand = self.cfg.getliteral(cfgsect, 'initial-candidate', [])
        self.validate_cand = self.cfg.getliteral(cfgsect, 'model-validate', [])

        if self.cfg.getpath(cfgsect, 'base-sim-location', None):
            # Read the base simulation data and prepare it for plotting
            # If this does not exist, then extrapolate the first iteration data 
            pass

        match suffix:
            case 'online':                
                intg.opt_type = suffix

                self._toptstart = self.cfg.getfloat(cfgostat, 'tstart', intg.tstart)     
                self._toptend = self.cfg.getfloat(cfgostat, 'tend', intg.tend)     
                self._tend = intg.tend     

                self.noptiters_max = ((self._toptend - intg.tcurr)/intg._dt
                                    //(skip_n + last_n))
                self.index_name = 'tcurr'
            case 'onfline':
                intg.opt_type = suffix
                self.noptiters_max = self.cfg.getint(cfgsect, 'iterations', 100)
                intg.offline_optimisation_complete = False

                self.index_name = 'iteration'
            case 'offline':
                raise NotImplementedError(f'{suffix} not implemented.')
            case _:
                raise ValueError('Not a valid suffix')

        if self.rank == self.root:

            self.KG_limit = self.cfg.getint(cfgsect, 
                'KG-limit', int(np.sqrt(self.noptiters_max)))

            self.EI_limit = self.cfg.getint(cfgsect, 
                'EI-limit', int(3*np.sqrt(self.noptiters_max)))

            self.PM_limit = self.cfg.getint(cfgsect, 
                'PM-limit', int(self.noptiters_max/2))

            print(  f"KG limit: {self.KG_limit}, ",
                    f"EI limit: {self.EI_limit}, ",
                    f"PM limit: {self.PM_limit}",
                    f"stop limit: {self.noptiters_max}",
                )

            import torch
            self.torch = torch

            seed = self.cfg.getint(cfgsect, 'seed', 0)

            self.torch.manual_seed(seed)
            self.torch.random.manual_seed(seed)

            # Stress test with gpu too, compare timings
            be = self.cfg.get(cfgsect, 'botorch-backend', 'cpu')
            pr = self.cfg.get(cfgsect, 'botorch-precision', 'double')

            if pr == 'double':
                self.torch_kwargs = {'dtype': torch.float64,
                                    'device': torch.device(be)}
            elif pr == 'single':
                self.torch_kwargs = {'dtype': torch.float32,
                                    'device': torch.device(be)}
            else:
                raise ValueError('Not a valid dtype')                                    

            bnds_init = list(zip(*self.cfg.getliteral(cfgsect, 'bounds')))
            hard_bnds = list(zip(*self.cfg.getliteral(cfgsect, 'hard-bounds')))
            self._bnds = self.torch.tensor(bnds_init, **self.torch_kwargs)
            self._hbnds = self.torch.tensor(hard_bnds, **self.torch_kwargs)

            self.cost_plot = self.cfg.get(cfgsect, 'cost-plot', 'cost.png')
            self.cumm_plot = self.cfg.get(cfgsect, 'cumm-plot', 'cumm-cost.png')
            self.speedup_plot = self.cfg.get(cfgsect, 'speedup-plot', 'speedup.png')
            self.outf = self.cfg.get(cfgsect, 'history'  , 'bayesopt.csv')

            self.pd_opt = pd.DataFrame(columns=[
                'test-candidate', 'test-m', 'test-s',
                'next-candidate', 'next-m', 'next-s',
                'best-candidate', 'best-m', 'best-s',
                'type', 'bounds-size',   
                ])
    #        arrays = [
    #                    ["test",      "test", "test", 
    #                     "next",      "next", "next", 
    #                     "best",      "best", "best", ],
    #                    ["candidate", "mean", "std" , 
    #                     "candidate", "mean", "std" , 
    #                     "candidate", "mean", "std" , ],
    #                    ]
    #        index = pd.MultiIndex.from_arrays(arrays, names=["candidate-type", "values"])
    #        print(index)

        intg.candidate = {}     # This will be used by the next plugin in line

    def __call__(self, intg):

        if not intg.reset_opt_stats:
            return

        self.check_offline_optimisation_status(intg)

        if self.rank == self.root:
            opt_time_start = perf_counter()

            if intg.opt_type == 'online' and intg.bad_sim:
                if np.isnan(intg.opt_cost_mean) and self.pd_opt.empty:
                    raise ValueError("Initial configuration must be working.")
#                tcurr = self.pd_opt.loc[self.pd_opt.index[-1], ['tcurr']]
                tcurr = self.pd_opt.iloc[-1, self.pd_opt.columns.get_loc('tcurr')]
            else:
                tcurr = intg.tcurr

            t1 =  pd.DataFrame({
                'test-candidate': [self._preprocess_candidate(
                                    *self.candidate_from_intg(intg))],
                'test-m': [intg.opt_cost_mean], 
                'test-s': [intg.opt_cost_std],
                })

            if self.init_cand != []:
                t1['next-candidate'] = [self.init_cand.pop(0)]
                t1['next-m'] = [np.nan]
                t1['next-s'] = [np.nan]
                t1['type'] = 'initial-testing'

            elif (len(self.pd_opt.index)<(self.PM_limit+len(self.validate_cand))
                  or intg.bad_sim):

                self.add_to_model(*self.process_raw(t1))
                t1['best-candidate'], t1['best-m'], t1['best-s'] = self.best_from_model()
                t1['bounds-size'] = self.bounds_size

                if self.validate_cand != []:
                    for i, c in enumerate(self.validate_cand):
                        tensored_c = self.torch.tensor([*list(c)], **self.torch_kwargs)
                        (t1[f'check-candidate-{i}'], 
                         t1[f'check-m-{i}'], 
                         t1[f'check-s-{i}']) = self.substitute_in_model(tensored_c)

                # Check if optimisation is performing alright
                if intg.bad_sim:
                    t1['next-candidate'] = t1['best-candidate']
                    t1['next-m'], t1['next-s'] = t1['best-m'], t1['best-s']
                    t1['type'] = 'PM fallback'

                elif (0.9*self.pd_opt['test-m'].min()) > t1['best-m'].tail(1).min():
                    t1['next-candidate'] = t1['best-candidate']
                    t1['next-m'], t1['next-s'] = t1['best-m'], t1['best-s']
                    t1['type'] = 'PM best-check'
                elif 0 > (t1['best-m'] - 2*t1['best-s']).tail(1).min():
                    t1['next-candidate'] = t1['best-candidate']
                    t1['next-m'], t1['next-s'] = t1['best-m'], t1['best-s']
                    t1['type'] = 'PM model-check'

                # Continue if no issues with the model or the best candidate
                elif (len(self.pd_opt.index))<self.KG_limit:
                    t1['next-candidate'], t1['next-m'], t1['next-s'] = self.next_from_model('KG')
                    t1['type'] = 'KG explore'

                elif (len(self.pd_opt.index))<self.EI_limit:
                    t1['next-candidate'], t1['next-m'], t1['next-s'] = self.next_from_model('EI')
                    t1['type'] = 'EI explore'

                elif (len(self.pd_opt.index))<self.PM_limit:
                    t1['next-candidate'] = t1['best-candidate']
                    t1['next-m'], t1['next-s'] = t1['best-m'], t1['best-s']
                    t1['type'] = 'PM exploit'

                elif self.validate_cand != []:
                    val_cand_tensor  = self.torch.tensor([*list(self.validate_cand.pop())], **self.torch_kwargs)
                    t1['next-candidate'], t1['next-m'], t1['next-s'] = self.substitute_in_model(val_cand_tensor)
                    t1['type'] = 'test-all-validation-candidates'

                else:
                    raise ValueError("No more candidates to test.")

            # Finally, use the best tested working candidate in the end
            else:

                t1['next-candidate'], t1['next-m'], t1['next-s'] = self.best_tested_from_dataframe()

#                t1['next-candidate'] = (self.pd_opt[self.pd_opt['test-m'].reset_index(drop=True) 
#                                                 == self.pd_opt['test-m'].min()]['test-candidate']).reset_index(drop=True)

#                t1['next-m'] = (self.pd_opt[self.pd_opt['test-m'].reset_index(drop=True) 
#                                == self.pd_opt['test-m'].min()]['test-m']
#                                .reset_index(drop=True))

#                t1['next-s'] = (self.pd_opt[self.pd_opt['test-m'].reset_index(drop=True) 
#                                == self.pd_opt['test-m'].min()]['test-s'].
#                                reset_index(drop=True))
                t1['type'] = 'stop'

            t1['opt-time'] = perf_counter() - opt_time_start

            match intg.opt_type:
                case 'online':
                    t1[self.index_name] = tcurr                
                case 'onfline':
                    t1[self.index_name] = len(self.pd_opt.index)+1
                case _:
                    raise ValueError('Not a valid opt_type')

            if intg.opt_type == 'online':
                t1['cumm-compute-time'] = intg.pseudointegrator._compute_time

            # Add all the data collected into the main dataframe
            self.pd_opt = pd.concat([self.pd_opt, t1], ignore_index=True)

            self.pd_opt['repetition'] = (self.pd_opt
                                         .groupby(['test-candidate'])
                                         .cumcount()+1)

            #with pd.option_context( 'display.max_rows'         , None, 
            #                        'display.max_columns'      , None,
            #                        'display.precision'        , 3,
            #                        'display.expand_frame_repr', False,
            #                        'display.max_colwidth'     , 100):
            #    print(self.pd_opt)

            self.pd_opt.to_csv(self.outf, index=False)

            if intg.opt_type == 'online':
                self.plot_normalised_cummulative_cost(intg)
                self.plot_overall_speedup(intg)

            self.plot_normalised_cost(intg)

            intg.candidate = self._postprocess_ccandidate(list(t1['next-candidate'])[0])
        intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def check_offline_optimisation_status(self, intg):
        if (self.rank == self.root) and (len(self.pd_opt.index) > self.noptiters_max):
            intg.offline_optimisation_complete = True
        else:
            intg.offline_optimisation_complete = False

        intg.offline_optimisation_complete = self.comm.bcast(
                                           intg.offline_optimisation_complete, 
                                           root = self.root)

    def process_raw(self, t1):

        # Process all optimisables
        tX  = np.array(list(self.pd_opt['test-candidate'])
                        +[list(t1['test-candidate'])[0]])
        tY  = np.array(list(self.pd_opt['test-m'])
                        +[list(t1['test-m'])[0]]).reshape(-1,1)
        tYv = np.array(list(self.pd_opt['test-s'])
                        +[list(t1['test-s'])[0]])**2

        # If NaN, then replace with twice the data of worst working candidate
        tY[np.isnan(tY)] = 2*np.nanmax(tY)
        tYv[np.isnan(tYv)] = 2*np.nanmax(tYv)
        tYv = tYv.reshape(-1,1)

        return tX, tY, tYv

    def best_tested_from_dataframe(self):
        """ 
            Return the best tested candidate and coresponding stats (mean, std)
        """
        grouped_pd_opt = self.pd_opt.groupby('test-candidate')

        candidate = (grouped_pd_opt.mean(numeric_only = True)['test-m']
                                   .idxmin())

        mean      = (grouped_pd_opt.mean(numeric_only = True)['test-m']
                                   .min())
        std = (grouped_pd_opt.std(numeric_only = True)
                             .fillna(grouped_pd_opt.last())
                             .at[candidate,'test-s'])

        return [candidate], mean, std

    def plot_normalised_cost(self, intg):
        """Plot cost function statistics.
                onfline: wrt number of iterations.
                online: wrt current time. 

            Normalise all of the below plots with the first iteration value.
            Plot the cost functions for the tested candidate.
            Plot the cost [mean, ub, lb] for predicted best candidate.

        """

        base = float(self.pd_opt.at[0, 'test-m'])

        cost_fig, cost_ax = plt.subplots(1,1, figsize = (8,8))

        cost_ax.set_title(f'Online optimisation \n base cost: {str(base)}')
        cost_ax.set_xlabel(self.index_name)
        cost_ax.set_ylabel('Base-normalised cost')

        cost_ax.axhline(y = 1, color = 'grey', linestyle = ':')
        
#            cost_ax.plot(self.pd_opt[self.index_name], 
#                        self.pd_opt['best-m']/base, 
#                        linestyle = ':', color = 'green',
#                        label = 'Predicted best mean cost', 
#                        )

#            cost_ax.fill_between(
#                    self.pd_opt[self.index_name], 
#                    (self.pd_opt['best-m']-2*self.pd_opt['best-s'])/base, 
#                    (self.pd_opt['best-m']+2*self.pd_opt['best-s'])/base, 
#                    color = 'green', alpha = 0.1, 
#                    label = 'Predicted best candidate''s confidence in cost',
#                        )

        cost_ax.scatter(self.pd_opt[self.index_name], 
                     self.pd_opt['test-m']/base, 
                     color = 'blue', marker = '*',
                     label = 'Tested candidate mean cost', 
                    )

        tested_best = (self.pd_opt['test-m']
                       .expanding().min()
                       .reset_index(drop=True))

        cost_ax.plot(self.pd_opt[self.index_name], 
                     tested_best/base,
                     linestyle = ':', color = 'blue',
                     label = 'Best tested cost', 
                    )
        # 
        cummin_loc = list(self.pd_opt['test-m']
                          .expanding().apply(lambda x: x.idxmin()).astype(int))

        tested_best_std = (self.pd_opt.loc[cummin_loc]['test-s']
                           .reset_index(drop=True))
        cost_ax.fill_between(
                self.pd_opt[self.index_name], 
                (tested_best-2*tested_best_std)/base, 
                (tested_best+2*tested_best_std)/base, 
                color = 'blue', alpha = 0.04, 
                label = 'Tested best candidate''s confidence in cost',
                    )

        cost_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt['next-m']/base,
                     linestyle = ':', color = 'green',
                     label = 'Best next cost', 
                    )

        next_index_add = intg._dt if intg.opt_type == 'online' else 1

        cost_ax.fill_between(
                self.pd_opt[self.index_name] + next_index_add, 
                (self.pd_opt['next-m']-2*self.pd_opt['next-s'])/base, 
                (self.pd_opt['next-m']+2*self.pd_opt['next-s'])/base, 
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

        proj = 1 + self.pd_opt.index - (self.pd_opt['test-m']
                                        .isna()
                                        .expanding().sum())

        base = (self.pd_opt.at[0, 'cumm-compute-time']*proj)

        cumm_ax.axhline(y = 1, color = 'grey', linestyle = ':',)

        ref = float(self.pd_opt.at[0, 'test-m'])

        if (len(self.pd_opt.index))<self.PM_limit:
            m = float(self.pd_opt[
                self.pd_opt['test-m'].reset_index(drop=True)
                == self.pd_opt['test-m'].min()
                ]['test-m']
                .reset_index(drop=True))

            s = float(self.pd_opt[
                self.pd_opt['test-m'].reset_index(drop=True) 
                == self.pd_opt['test-m'].min()
                ]['test-s']
                .reset_index(drop=True))

            cumm_ax.axhline(y = ref/m, 
                            linestyle = ':', color = 'green', 
                            )

            cumm_ax.axhspan(ref/(m-2*s), ref/(m+2*s), 
                            color = 'green', alpha = 0.1, 
                            label = 'Possible best tested speed-up'
                            )
        else:        

            m1 = self.pd_opt.at[self.pd_opt.last_valid_index(), 'next-m']
            s1 = self.pd_opt.at[self.pd_opt.last_valid_index(), 'next-s']

            cumm_ax.axhline(y = ref/m1, 
                            linestyle = ':', color = 'green', 
                            )

            cumm_ax.axhspan(ref/(m1-2*s1), ref/(m1+2*s1), 
                            color = 'green', alpha = 0.1, 
                            label = 'Aiming speed-up',
                            )

        cumm_ax.plot(self.pd_opt[self.index_name], 
                     base/self.pd_opt['cumm-compute-time'],
                     linestyle = '-', color = 'g', marker = '*',
                     label = 'compute-time speedup',
                    )

        cumm_ax.plot(self.pd_opt[self.index_name], 
                     base/(self.pd_opt['cumm-compute-time'] + 
                           self.pd_opt['opt-time'].expanding().sum()),
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

    def plot_normalised_cummulative_cost(self, intg):
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


        proj = self.pd_opt.index - (self.pd_opt['test-m']
                                    .isna().expanding().sum() + 1)
        
        cumm_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt.at[0, 'cumm-compute-time']*proj,
                     linestyle = '--', color = 'grey',
                     label = 'projected compute-time',
                    )

        cumm_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt['cumm-compute-time'] + 
                     self.pd_opt['opt-time'].expanding().sum(),
                     linestyle = '-', color = 'blue',
                     label = 'actual cost (compute + optimisation)',
                    )

        cumm_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt['cumm-compute-time'],
                     linestyle = '-', color = 'green',
                     label = 'compute-cost',
                    )

        cumm_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt['opt-time'].expanding().sum(),
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
        # Plot vertical lines to indicate the switch in search type
        if len(self.pd_opt.index)>self.KG_limit:
            ax.axvline(
                x = float(self.pd_opt.at[self.KG_limit, self.index_name]),
                color = 'red', linestyle = '-.',
                label = 'KG-EI transition', 
                )
            
        if len(self.pd_opt.index)>self.EI_limit:
            ax.axvline(
                x = float(self.pd_opt.at[self.EI_limit, self.index_name]),
                color = 'orange', linestyle = '-.',
                label = 'EI-PM transition', 
                )

        if len(self.pd_opt.index)>self.PM_limit:
            ax.axvline(
                x = float(self.pd_opt.at[self.PM_limit, self.index_name]),
                color = 'yellow', linestyle = '-.',
                label = 'Optimisation turned off', 
                )

    def store_validation_from_model(self, t1, i, test_case):
        """ Sample a validation point from the model
            Store the results into the dataframe

        Args:
            t1 (pandas.DataFrame): Temporary dataframe to add the model data into
            i (int): Sample number
            test_case (list[Float]): _description_
        """

            #t1 = self.store_validation_from_model(t1, 1, (1, 1, 1, 1))

        test = self.torch.tensor([test_case], **self.torch_kwargs)
        _, t1[f'test-{i}-m'], t1[f'test-{i}-s'], = self.substitute_in_model(test)
        return t1

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
        from botorch.models.transforms import Standardize, Log, Normalize, ChainedOutcomeTransform

        self.bounds = tX # Set bounds on the basis of 
                            # training data 
                            # given soft-bounds
                            # given hard-bounds

        self.normalise = Normalize(d=tX.shape[1], bounds=self.bounds)
        self.standardise = Standardize(m=1)

        self.trans = ChainedOutcomeTransform(
#            log = Log(),                # First take log of cost
            stan = Standardize(m=1),    # Then standardise
            )

        self.norm_X = self.normalise.transform(
            self.torch.tensor(tX , **self.torch_kwargs))
        stan_Y, stan_Yvar = self.trans.forward(
            self.torch.tensor(tY , **self.torch_kwargs),
            self.torch.tensor(tYv, **self.torch_kwargs))

        self.model = FixedNoiseGP(train_X = self.norm_X, 
                                  train_Y = stan_Y, 
                                  train_Yvar = stan_Yvar)

        mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, 
                                         model = self.model)

        mll = mll.to(**self.torch_kwargs)                      
        fit_model(mll)

    @property
    def bounds(self):
        return self._bnds

    @bounds.setter
    def bounds(self, tX:np.ndarray):

        # Create happening region 
        # a cube of length 2*radius around the best points
        radius = int(np.ceil(np.sqrt(tX.shape[0])))
        best_cands = tX[tX.argsort(axis=0)[:, -2] <= radius, :]
        means, stds = np.mean(best_cands, axis=0), np.std( best_cands, axis=0)
        hr = self.torch.tensor(np.array([means - self.bnds_var * stds,
                                         means + self.bnds_var * stds]), 
                               **self.torch_kwargs)

        # Increase bounds to include happening region
        l_bnds_inc_loc = self._bnds[0, :] > hr[0, :]
        u_bnds_inc_loc = self._bnds[1, :] < hr[1, :]
        
        self._bnds[0, l_bnds_inc_loc] = hr[0, l_bnds_inc_loc]
        self._bnds[1, u_bnds_inc_loc] = hr[1, u_bnds_inc_loc]
        self._bnds[0, self._bnds[0, :] < 0] = 0

        # Restrict the bounds to hardbounds region
        self._bnds[0, :] = self.torch.maximum(self._bnds[0, :], self._hbnds[0, :])
        self._bnds[1, :] = self.torch.minimum(self._bnds[1, :], self._hbnds[1, :])

    @property
    def bounds_size(self):
        return np.prod((self._bnds[1,:]-self._bnds[0,:]).detach().cpu().numpy())
        
    def candidate_from_intg(self, intg):
        unprocessed = []
        for optimisable in self.optimisables:
            match optimisable:
                case 'csteps':
                    unprocessed.append(intg.pseudointegrator.csteps)
                case 'pseudo-dt-max':
                    unprocessed.append(intg.pseudointegrator.pintg.Δτᴹ)
                case 'pseudo-dt-fact':
                    unprocessed.append(intg.pseudointegrator.dtauf)
                case _:
                    raise ValueError(f"Unrecognised optimisable: {optimisable}")
        return unprocessed

    def _preprocess_candidate(self, *args):
        '''
            Processing must start with V-cycle
        '''

        csteps = args[0]
        self.depth = (len(csteps)-1)//2 -1
        return csteps[1], csteps[self.depth+1], csteps[-2], csteps[-1], *args[1:]

    def _postprocess_ccandidate(self, ccandidate):
        post_processed = {}

        for optimisable in self.optimisables:
            match optimisable:
                case 'csteps':
                    post_processed['csteps'] = self._postprocess_ccsteps(ccandidate[:4])
                case 'pseudo-dt-max':
                    post_processed['pseudo-dt-max'] = ccandidate[4]
                case 'pseudo-dt-fact':
                    post_processed['pseudo-dt-fact'] = ccandidate[5]
                case _:
                    raise ValueError(f"Unrecognised optimisable {optimisable}")

        return post_processed

    def _postprocess_ccsteps(self, ccsteps):
        return (*((ccsteps[0],)*(self.depth+1)), ccsteps[1], 
                *((ccsteps[2],)*self.depth    ), ccsteps[3])

    def best_from_model(self, *, raw_samples=1024, num_restarts=10):

        from botorch.acquisition import PosteriorMean

        _acquisition_function = PosteriorMean(
            self.model, 
            maximize = False,
            )

        X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
        X_b    = self.normalise.untransform(X_cand)

        return self.substitute_in_model(X_b)

    def next_from_model(self, type = 'EI', *, 
                        samples=1024, restarts=10,):
        """ Using model, optimise acquisition function and return next candidate

        Args:
            type (str): Acquisition function acronym. Defaults to 'EI'.
            raw_samples (int, optional): _description_. Defaults to 1024.
            num_restarts (int, optional): _description_. Defaults to 10.

        Raises:
            ValueError: _description_

        Returns:
            tuple[list[float], float, float]: Candidate, expected mean and var
        """

        negw = self.torch.tensor([-1.0], **self.torch_kwargs)

        if type == 'KG':

            from botorch.acquisition           import qKnowledgeGradient
            from botorch.acquisition.objective import ScalarizedPosteriorTransform
            
            _acquisition_function = qKnowledgeGradient(
                self.model, 
                posterior_transform = ScalarizedPosteriorTransform(weights=negw),
                num_fantasies       = 128,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'EI':

            from botorch.acquisition           import qNoisyExpectedImprovement
            from botorch.acquisition.objective import ScalarizedPosteriorTransform
            
            _acquisition_function = qNoisyExpectedImprovement(
                self.model, self.norm_X,
                posterior_transform = ScalarizedPosteriorTransform(weights=negw),
                prune_baseline = True,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)
        else:
            raise ValueError(f'next_type {type} not recognised')

        return self.substitute_in_model(X_b)

    def optimise(self, _acquisition_function, raw_samples=4096, num_restarts=100):
        """ Returns a transformed attached candidate which minimises the acquisition function
        """

        from botorch.optim.optimize import optimize_acqf

        X_cand, _ = optimize_acqf(
            acq_function = _acquisition_function,
            bounds       = self.normalise.transform(self._bnds),
            q            = 1,
            num_restarts = num_restarts,
            raw_samples  = raw_samples,
        )

        return X_cand

    def substitute_in_model(self, X_b):
        """ Accept an untransformed candidate into model. 
            Get an untransformed output.

        Args:
            X_b: Untransformed attached tensor of shape (1,4)

        Returns:
            X_best: tuple[float64, float64, float64, float64]
            Y_mean: float64
            Y_std : float64
        """

        X_best = self.normalise.transform(X_b)
        Y_low, Y_upp = self.model.posterior(X_best).mvn.confidence_region()
        Y_avg = self.model.posterior(X_best).mvn.mean

        Y_m = self.trans.untransform(Y_avg)[0].squeeze().detach().cpu().numpy()
        Y_l = self.trans.untransform(Y_low)[0].squeeze().detach().cpu().numpy()
        Y_u = self.trans.untransform(Y_upp)[0].squeeze().detach().cpu().numpy()
        Y_std = (Y_u - Y_l)/4

        return [tuple(X_b.squeeze().detach().cpu().numpy())], Y_m, Y_std
