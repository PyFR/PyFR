from time import perf_counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root

class BayesianOptimisationPlugin(BasePlugin):
    """ Bayesian Optimisation applied to PyFR.

        The configuration file for this plugin should contain the following:
        
        cost-plot    = ./bo_cost-plot.png
        cumm-plot    = ./bo_cumm-plot.png
        speedup-plot = ./bo_speedup-plot.png
        history      = ./bo_history.csv
        seed = 0

        KG-limit   = 5
        EI-limit   = 10
        PM-limit   = 15

        botorch-backend   = cuda
        botorch-precision = double

        optimisables = < a list of strings, where strings must be the following>
                       < 'cstep:0','cstep:1','cstep:2','cstep:3',
                         'pseudo-dt-max'>

        bounds      = [(0.0125, 0.0150)]
        hard-bounds = [(0.0125, 0.05)]

    """

    name = 'bayesian_optimisation'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix, **data):

        super().__init__(intg, cfgsect, suffix)

        cfgostat = 'soln-plugin-optimisation_stats'
        skip_n = self.cfg.getint(cfgostat, 'skip-first-n', 10)     
        last_n = self.cfg.getint(cfgostat, 'capture-last-n', 40)

        self.optimisables = self.cfg.getliteral(cfgsect, 'optimisables')

        self.columns = [
            *[f't-{i}' for i in range(len(self.optimisables))],'t-m', 't-s', 
            *[f'n-{i}' for i in range(len(self.optimisables))],'n-m', 'n-s', 
            *[f'b-{i}' for i in range(len(self.optimisables))],'b-m', 'b-s',
            'bounds-size', 
            'opt-time', 'tcurr', 'cumm-compute-time', 
            'repetition', 'LooCV',
            ] 

        self.bnds_var = self.cfg.getfloat(cfgsect, 'bounds-variability', 2)

        self.KG_limit = self.cfg.getint(cfgsect, 'KG-limit')
        self.EI_limit = self.cfg.getint(cfgsect, 'EI-limit')
        self.PM_limit = self.cfg.getint(cfgsect, 'PM-limit')

        self.comm, self.rank, self.root = get_comm_rank_root()

        # Read the initial candidate and the validation candidate
        #   This is a list of tuple of 4 floats

        if self.cfg.getpath(cfgsect, 'base-sim-location', None):
            # Read the base simulation data and prepare it for plotting
            # If this does not exist, then extrapolate the first iteration data 
            pass

        if suffix == 'online':
            intg.opt_type = 'online'

            self._toptstart = self.cfg.getfloat(cfgostat, 'tstart', intg.tstart)     
            self._toptend = self.cfg.getfloat(cfgostat, 'tend', intg.tend)     
            self._tend = intg.tend     

            self.noptiters_max = ((self._toptend - intg.tcurr)/intg._dt
                                //(skip_n + last_n))
            self.index_name = 'tcurr'
        elif suffix == 'onfline':
            intg.opt_type = 'onfline'
            self.noptiters_max = self.cfg.getint(cfgsect, 'iterations', 100)
            intg.offline_optimisation_complete = False
            self.index_name = 'iteration'
        elif suffix == 'offline':
            raise NotImplementedError(f'offline not implemented.')
        else:
            raise ValueError('Invalid suffix')

        print(  f"KG limit: {self.KG_limit}, ",
                f"EI limit: {self.EI_limit}, ",
                f"PM limit: {self.PM_limit}",
                f"stop limit: {self.noptiters_max}",
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

            bnds_init = list(zip(*self.cfg.getliteral(cfgsect, 'bounds')))
            hard_bnds = list(zip(*self.cfg.getliteral(cfgsect, 'hard-bounds')))
            self._bnds = self.torch.tensor(bnds_init, **self.torch_kwargs)
            self._hbnds = self.torch.tensor(hard_bnds, **self.torch_kwargs)

            self.cost_plot = self.cfg.get(cfgsect, 'cost-plot', None)
            self.cumm_plot = self.cfg.get(cfgsect, 'cumm-plot', None)
            self.speedup_plot = self.cfg.get(cfgsect, 'speedup-plot', None)
            self.outf = self.cfg.get(cfgsect, 'history'  , 'bayesopt.csv')

            self.validate = self.cfg.getliteral(cfgsect, 'validate',[])

        self.deserialise(intg, data)

    def __call__(self, intg):

        if not intg.reset_opt_stats:
            return

        self.check_offline_optimisation_status(intg)

        self.loocv_error = 100

        if self.rank == self.root:
            print(intg.tcurr)
            opt_time_start = perf_counter()

            if intg.opt_type == 'online' and intg.bad_sim:
                if np.isnan(intg.opt_cost_mean) and self.pd_opt.empty:
                    raise ValueError("Initial configuration must be working.")
#                tcurr = self.pd_opt.loc[self.pd_opt.index[-1], ['tcurr']]
                tcurr = self.pd_opt.iloc[-1, self.pd_opt.columns.get_loc('tcurr')]
            else:
                tcurr = intg.tcurr

            t1 =  pd.DataFrame({
                **{f't-{i}': [val] for i, val in enumerate(self.candidate_from_intg(intg))},
                't-m': [intg.opt_cost_mean], 
                't-s': [intg.opt_cost_std], 
                },)
            
            if not self.validate == [] : 
                next_candidate = self.validate.pop()

                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val

            elif ((len(self.pd_opt.index)<(self.PM_limit)) or intg.bad_sim):

                self.add_to_model(*self.process_raw(t1))
                best_candidate, t1['b-m'], t1['b-s'] = self.best_from_model()

                for i, val in enumerate(best_candidate):
                    t1[f'b-{i}'] = val

                t1['bounds-size'] = self.bounds_size

                # Check if optimisation is performing alright
                if (intg.bad_sim or
                    (0.9*self.pd_opt['t-m'].min()) > t1['b-m'].tail(1).min() or
                    0 > (t1['b-m'] - 2*t1['b-s']).tail(1).min() or
                    self.pd_opt.empty):

                    for i, val in enumerate(best_candidate):
                        t1[f'n-{i}'] = val
                        t1['n-m'], t1['n-s'] = t1['b-m'], t1['b-s']

                    if intg.bad_sim:
                        print("Bad simulation, using best candidate.")
                    elif (0.9*self.pd_opt['t-m'].min()) > t1['b-m'].tail(1).min():
                        print("Best candidate wall-time is lesser than 0.9 times the previous best.")
                    elif 0 > (t1['b-m'] - 2*t1['b-s']).tail(1).min():
                        print("Best candidate is worse than 2 sigma of the previous best.")
                    else:
                        print("No previous optimisation data. Using best candidate which is random enough.")

                # Continue (if no issues) with the model or the best candidate

                elif (len(self.pd_opt.index))<self.KG_limit:
                    next_candidate, t1['n-m'], t1['n-s'] = self.next_from_model('KG')
                    for i, val in enumerate(next_candidate):
                        t1[f'n-{i}'] = val
                    print(f"KG: {next_candidate}")

                elif (len(self.pd_opt.index))<self.EI_limit:
                    next_candidate, t1['n-m'], t1['n-s'] = self.next_from_model('EI')
                    for i, val in enumerate(next_candidate):
                        t1[f'n-{i}'] = val
                    print(f"EI: {next_candidate}")

                elif (len(self.pd_opt.index))<self.PM_limit:
                    for i, val in enumerate(best_candidate):
                        t1[f'n-{i}'] = val
                    t1['n-m'], t1['n-s'] = t1['b-m'], t1['b-s']
                    print(f"Best: {best_candidate}")
                else:
                    raise ValueError("No more candidates to test.")

            # Finally, use the best tested working candidate in the end
            else:
                next_candidate, t1['n-m'], t1['n-s'] = self.best_tested_from_dataframe()
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                
                # If offline optimisation, then abort the simulation at this point. 
                if intg.opt_type == 'onfline':
                    intg.abort = True
                    
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

            t1['LooCV'] = self.loocv_error

            # Add all the data collected into the main dataframe
            self.pd_opt = pd.concat([self.pd_opt, t1], ignore_index=True)

            test_cols = list(filter(lambda x: x.startswith('t-') or x.startswith('t-'), t1.columns))
            self.pd_opt['repetition'] = (self.pd_opt
                                        .groupby(test_cols)
                                        .cumcount()+1)

            self.pd_opt.to_csv(self.outf, index=False)

#            if intg.opt_type == 'online' and self.cumm_plot and self.speedup_plot:
#                self.plot_normalised_cummulative_cost()
#                self.plot_overall_speedup(intg)
#
#            if self.cost_plot:
#                self.plot_normalised_cost(intg)

            next_cols = list(filter(lambda x: x.startswith('n-') and x[2:].isdigit(), self.pd_opt.columns))

            print(f"{t1[next_cols] = }") 

            intg.candidate = self._postprocess_ccandidate(list(t1[next_cols].values)[0])

        print(f"{intg.candidate = }") 
        intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def serialise(self, intg):
        if self.rank == self.root:
            return {'pd_opt':self.pd_opt.to_numpy(dtype=np.float64),
                    'bounds':self.bounds,
                    }

    def deserialise(self, intg, data):

        intg.candidate = {}
        self.depth = self.cfg.getint('solver', 'order', 0)

        if self.rank == self.root:

            if bool(data):
                self.pd_opt = pd.DataFrame(data['pd_opt'], columns=self.columns)
                self.bounds = data['bounds']

                # Set the candidate to the next candidate if optimisation was being performed in the previous run                
                next_cols = list(filter(lambda x: x.startswith('n-') and x[2:].isdigit(), self.pd_opt.columns))
                next_candidate = self.pd_opt[next_cols].tail(1).to_numpy(dtype=np.float64)[0]
                intg.candidate = self._postprocess_ccandidate(next_candidate)
            else:
                self.pd_opt = pd.DataFrame(columns=self.columns)
            
        if (self.rank == self.root) and len(intg.candidate)>0:
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
        test_cols = list(filter(lambda x: x.startswith('t-') and x[2:].isdigit(), t1.columns))

        # tX  = np.vstack((self.pd_opt[test_cols].to_numpy(dtype=np.float64), 
        #                           t1[test_cols].to_numpy(dtype=np.float64)))
        # tY  = np.array(list(self.pd_opt['t-m'])+[list(t1['t-m'])[0]]).reshape(-1,1)
        # tYv = np.array(list(self.pd_opt['t-s'])+[list(t1['t-s'])[0]])**2

        tX  = pd.concat([self.pd_opt[test_cols], t1[test_cols]]     , axis=0, ignore_index=True, sort=False).astype(np.float64).to_numpy()
        tY  = pd.concat([self.pd_opt['t-m']    , t1['t-m'].iloc[:1]], axis=0, ignore_index=True, sort=False).to_numpy().reshape(-1, 1)
        tYv = pd.concat([self.pd_opt['t-s']    , t1['t-s'].iloc[:1]], axis=0, ignore_index=True, sort=False).to_numpy().reshape(-1, 1) ** 2

        # If NaN, then replace with twice the data of worst working candidate
        tY[np.isnan(tY)] = 2*np.nanmax(tY)
        tYv[np.isnan(tYv)] = 2*np.nanmax(tYv)
        tYv = tYv.reshape(-1,1)

        return tX, tY, tYv

    def best_tested_from_dataframe(self):
        """ 
            Return the best tested candidate and coresponding stats (mean, std)
        """
        test_cols = list(filter(lambda x: x.startswith('t-') and x[2:].isdigit(), self.pd_opt.columns))
        grouped_pd_opt = self.pd_opt.groupby(test_cols)

        candidate = (grouped_pd_opt.mean(numeric_only = True)['t-m'].idxmin())
        mean = (grouped_pd_opt.mean(numeric_only = True)['t-m'].min())
        std = (grouped_pd_opt.std(numeric_only = True).fillna(grouped_pd_opt.last())
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

        base = float(self.pd_opt.at[0, 't-m'])

        cost_fig, cost_ax = plt.subplots(1,1, figsize = (8,8))

        cost_ax.set_title(f'Online optimisation \n base cost: {str(base)}')
        cost_ax.set_xlabel(self.index_name)
        cost_ax.set_ylabel('Base-normalised cost')

        cost_ax.axhline(y = 1, color = 'grey', linestyle = ':')
        
#            cost_ax.plot(self.pd_opt[self.index_name], 
#                        self.pd_opt['b-m']/base, 
#                        linestyle = ':', color = 'green',
#                        label = 'Predicted best mean cost', 
#                        )

#            cost_ax.fill_between(
#                    self.pd_opt[self.index_name], 
#                    (self.pd_opt['b-m']-2*self.pd_opt['b-s'])/base, 
#                    (self.pd_opt['b-m']+2*self.pd_opt['b-s'])/base, 
#                    color = 'green', alpha = 0.1, 
#                    label = 'Predicted best candidate''s confidence in cost',
#                        )

        cost_ax.scatter(self.pd_opt[self.index_name], 
                     self.pd_opt['t-m']/base, 
                     color = 'blue', marker = '*',
                     label = 'Tested candidate mean cost', 
                    )

        tested_best = (self.pd_opt['t-m']
                       .expanding().min()
                       .reset_index(drop=True))

        cost_ax.plot(self.pd_opt[self.index_name], 
                     tested_best/base,
                     linestyle = ':', color = 'blue',
                     label = 'Best tested cost', 
                    )
        # 
        cummin_loc = list(self.pd_opt['t-m']
                          .expanding().apply(lambda x: x.idxmin()).astype(int))

        tested_best_std = (self.pd_opt.loc[cummin_loc]['t-s']
                           .reset_index(drop=True))
        cost_ax.fill_between(
                self.pd_opt[self.index_name], 
                (tested_best-2*tested_best_std)/base, 
                (tested_best+2*tested_best_std)/base, 
                color = 'blue', alpha = 0.04, 
                label = 'Tested best candidate''s confidence in cost',
                    )

        cost_ax.plot(self.pd_opt[self.index_name], 
                     self.pd_opt['n-m']/base,
                     linestyle = ':', color = 'green',
                     label = 'Best next cost', 
                    )

        next_index_add = intg._dt if intg.opt_type == 'online' else 1

        cost_ax.fill_between(
                self.pd_opt[self.index_name] + next_index_add, 
                (self.pd_opt['n-m']-2*self.pd_opt['n-s'])/base, 
                (self.pd_opt['n-m']+2*self.pd_opt['n-s'])/base, 
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

        proj = 1 + self.pd_opt.index - (self.pd_opt['t-m']
                                        .isna()
                                        .expanding().sum())

        base = (self.pd_opt.at[0, 'cumm-compute-time']*proj)

        cumm_ax.axhline(y = 1, color = 'grey', linestyle = ':',)

        ref = float(self.pd_opt.at[0, 't-m'])

        if (len(self.pd_opt.index))<self.PM_limit:
            m = float(self.pd_opt[
                self.pd_opt['t-m'].reset_index(drop=True)
                == self.pd_opt['t-m'].min()]['t-m'].reset_index(drop=True))

            s = float(self.pd_opt[
                self.pd_opt['t-m'].reset_index(drop=True) == 
                self.pd_opt['t-m'].min()]['t-s'].reset_index(drop=True))

            cumm_ax.axhline(y = ref/m, 
                            linestyle = ':', color = 'green', 
                            )

            cumm_ax.axhspan(ref/(m-2*s), ref/(m+2*s), 
                            color = 'green', alpha = 0.1, 
                            label = 'Possible best tested speed-up'
                            )
        else:        

            m1 = self.pd_opt.at[self.pd_opt.last_valid_index(), 'n-m']
            s1 = self.pd_opt.at[self.pd_opt.last_valid_index(), 'n-s']

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

        proj = self.pd_opt.index - (self.pd_opt['t-m']
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
        limits = [(self.KG_limit, 'red', 'KG-EI transition'),
                (self.EI_limit, 'orange', 'EI-PM transition'),
                (self.PM_limit, 'yellow', 'Optimisation turned off')]

        for limit, color, label in limits:
            if len(self.pd_opt.index) > limit:
                ax.axvline(x=float(self.pd_opt.at[limit, self.index_name]),
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
        from botorch.models.transforms import Standardize, Normalize, ChainedOutcomeTransform

        self.expand_bounds(tX, tY) # Set bounds on the basis of training data 
                         # given soft-bounds and hard-bounds

        print(f'Bounds: {self._bnds}')

        self.normalise = Normalize(d=tX.shape[1], bounds=self._bnds)
        self.standardise = Standardize(m=1)

        self.trans = ChainedOutcomeTransform(
#            log = Log(),                # First take log of cost
            stan = Standardize(m=1),    # Then standardise
            )

        self._norm_X = self.normalise.transform(
            self.torch.tensor(tX , **self.torch_kwargs))
        stan_Y, stan_Yvar = self.trans.forward(
            self.torch.tensor(tY , **self.torch_kwargs),
            self.torch.tensor(tYv, **self.torch_kwargs))

        self.model = FixedNoiseGP(train_X = self._norm_X, 
                                  train_Y = stan_Y, 
                                  train_Yvar = stan_Yvar)

        mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, 
                                         model = self.model)

        mll = mll.to(**self.torch_kwargs)                      
        fit_model(mll)

        if (len(self.pd_opt.index))>=4:
            # Calculate LOOCV error only after the initialising 16 iterations
            # This is to avoid wasting time in the start
            #   when model is definitely not good enough
            
            self.loocv_error = self.cv_folds(self._norm_X, stan_Y, stan_Yvar).detach().cpu().numpy()
            print(self.cv_folds(self._norm_X, stan_Y, stan_Yvar).detach().cpu().numpy())

    @property
    def bounds(self):
        return self._bnds.detach().cpu().numpy()

    @bounds.setter
    def bounds(self, y):
        self._bnds = self.torch.tensor(y, **self.torch_kwargs)

    def expand_bounds(self, tX, tY):
        # Create happening region 
        # a cube of length 2*radius around the best points
        radius = int(np.ceil(np.sqrt(tX.shape[0])))
        best_cands = tX[tY.argsort(axis=0)[:,0] <= radius, :]
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
        self._bnds[0, :] = self.torch.max(self._bnds[0, :], self._hbnds[0, :])
        self._bnds[1, :] = self.torch.min(self._bnds[1, :], self._hbnds[1, :])

    @property
    def bounds_size(self):
        return np.prod((self._bnds[1,:]-self._bnds[0,:]).detach().cpu().numpy())
        
    def candidate_from_intg(self, intg):
        """ 
            Get candidate used in the last window from the integrator 
            Store the results into a dictionary mapping optimisable name to value
            Use the same optimisable name in modify_configuration plugin
        """
        unprocessed = []
        for optimisable in self.optimisables:
            match optimisable.split(':'):
                case ['cstep', i]:
                    csteps = self._preprocess_csteps4(intg.pseudointegrator.csteps)
                    unprocessed.append(csteps[int(i)])
                case ['pseudo-dt-max',]:
                    unprocessed.append(intg.pseudointegrator.pintg.Δτᴹ)
                case ['pseudo-dt-fact',]:
                    unprocessed.append(intg.pseudointegrator.dtauf)
                case _:
                    raise ValueError(f"Unrecognised optimisable: {optimisable}")
        return unprocessed

    def _postprocess_ccandidate(self, ccandidate):
        post_processed = {}
        for i, optimisable in enumerate(self.optimisables):
            if optimisable.startswith('cstep:') and optimisable[6:].isdigit():
                post_processed[optimisable] = ccandidate[i]
            elif optimisable == 'pseudo-dt-max':
                post_processed[optimisable] = ccandidate[i]
            elif optimisable == 'pseudo-dt-fact':
                post_processed[optimisable] = ccandidate[i]
            else:
                raise ValueError(f"Unrecognised optimisable {optimisable}")
        return post_processed

    def _preprocess_csteps4(self, csteps):
        return csteps[0], csteps[self.depth], csteps[-2], csteps[-1]
        
    def _preprocess_csteps2(self, csteps):
        return csteps[self.depth], csteps[-1]
        
    def _preprocess_csteps1(self, csteps):
        return csteps[-1],
        
    def best_from_model(self):

        from botorch.acquisition import PosteriorMean

        _acquisition_function = PosteriorMean(
            self.model, 
            maximize = False,
            )

        raw_samples  = 1024 if self.be == 'cpu' else 4096
        num_restarts =   10 if self.be == 'cpu' else   20

        X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
        X_b    = self.normalise.untransform(X_cand)

        return self.substitute_in_model(X_b)

    def next_from_model(self, type = 'EI'):
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
            
            samples   = 1024
            restarts  =   10
            fantasies =  128
            _acquisition_function = qKnowledgeGradient(
                self.model, 
                posterior_transform = ScalarizedPosteriorTransform(weights=negw),
                num_fantasies       = fantasies,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'EI':

            from botorch.acquisition           import qNoisyExpectedImprovement
            from botorch.acquisition.objective import ScalarizedPosteriorTransform

            samples   = 4096 if self.be == 'cuda' else 1024
            restarts  =   20 if self.be == 'cuda' else   10

            _acquisition_function = qNoisyExpectedImprovement(
                self.model, self._norm_X,
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

        XX = X_b.detach().cpu().squeeze().tolist()

        if isinstance(XX, float):
            return [XX], Y_m, Y_std
        else:
            return XX, Y_m, Y_std
        
    def cv_folds(self, train_X, train_Y, train_Yvar):

        from botorch.cross_validation import gen_loo_cv_folds

        cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)

        from botorch.cross_validation import batch_cross_validation
        from botorch.models import FixedNoiseGP
        from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        # instantiate and fit model
        cv_results = batch_cross_validation(
            model_cls=FixedNoiseGP,
            mll_cls=ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
        )

        posterior = cv_results.posterior
        mean = posterior.mean
        cv_error = ((cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()
        
        return cv_error
