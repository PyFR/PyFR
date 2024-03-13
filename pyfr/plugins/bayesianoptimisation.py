from time import perf_counter
import numpy as np
import pandas as pd

from pyfr.plugins.base import BaseSolnPlugin
from pyfr.mpiutil import get_comm_rank_root

class BayesianOptimisationPlugin(BaseSolnPlugin):
    """ Bayesian Optimisation applied to some optimisable parameters in PyFR.
        Currently, the optimisation is done using the BoTorch library.
        
        Parameters optimised:
            - multi-p cycle
            - max limit on Δτ
            - Multiplying factor of Δτ along multi-p levels

    """

    name = 'bayesian_optimisation'
    systems = ['*']
    formulations = ['dual']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix):

        super().__init__(intg, cfgsect, suffix)
        self.comm, self.rank, self.root = get_comm_rank_root()

        self.optimisables = self.cfg.getliteral(cfgsect, 'optimisables')
        self.bad_sim_multiplier = self.cfg.getfloat(cfgsect, 'bad-sim-multiplier', 2.0)
        self.columns_from_optimisables()

        self._nbcs = len(self.optimisables) # Number of best candidates to consider
        initialise_ref_cands = self.cfg.getfloat(cfgsect, 'initialise-reference', 2**(self._nbcs))

####################################################################################################################################################################################################################

        # Quickly get some optimum
        self._Ainit_lim  =     initialise_ref_cands # Initialise
        self._Binit_lim  = 1.5*initialise_ref_cands # Explore
        self._Cinit_lim  = 2.0*initialise_ref_cands # Exploit
        self._Dinit_lim  = 2.5*initialise_ref_cands # Explore/Exploit + expand

        # When to stop the offline simulation
        self._END_lim    = self.cfg.getfloat(cfgsect, 'stop-optimisation')
        self.force_abort = self.cfg.getbool( cfgsect, 'force-abort', False)

        self.mean_mult = self.cfg.getfloat(cfgsect, 'mean-multiplier', 0.1)
        self.std_mult  = self.cfg.getfloat(cfgsect, 'std-multiplier' , 1.0)

        intg.opt_type = self.opt_type = suffix
        if suffix == 'online':
            self.columns.append('tcurr')
            self.index_name = 'tcurr'
        elif suffix == 'onfline':
            self.columns.append('iteration')
            self.index_name = 'iteration'
            intg.offline_optimisation_complete = False
        else:
            raise ValueError('Invalid suffix')

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
            self.outf = self.cfg.get(cfgsect, 'history'  , 'bayesopt.csv')
            self.test = self.cfg.getliteral(cfgsect, 'test',[])

            # The latest tested candidate characteristics are determined by t1 dataframe
            # The following candidates determine what the next candidate should be like
            self.cand_phase, self.cand_train, self.cand_validate = 1, True, False # First candidate

        intg._no_more_optimisation = False
        intg.candidate = {}
        self.depth = self.cfg.getint('solver', 'order', 0)

        if self.rank == self.root:
            self.df_train = pd.DataFrame(columns=self.columns)

    def __call__(self, intg):

        # Skip optimisation untill opt-stats is reset
        if (not intg.reset_opt_stats) or intg._no_more_optimisation:
            return
    
        if self.rank == self.root:
            opt_time_start = perf_counter()

            if self.opt_type == 'online' and intg.bad_sim:
                if np.isnan(intg.opt_cost_mean) and self.df_train.empty:
                    raise ValueError("Start with a working base config.")
                tcurr = self.df_train.iloc[-1, self.df_train.columns.get_loc('tcurr')]
            else:
                tcurr = intg.tcurr

            # Convert last iteration data from intg to dataframe
            tested_candidate = self.candidate_from_intg(intg.pseudointegrator)

            t1 =  pd.DataFrame({
                **{f't-{i}': [val] for i, val in enumerate(tested_candidate)},
                't-m':         [intg.opt_cost_mean], 
                't-d':         [intg.opt_cost_std], 
                't-e':         [intg.opt_cost_sem], 
                'phase':       [self.cand_phase],           # |   
                'if-train':    [self.cand_train],       # |----> Latest tested candidate 
                'if-validate': [self.cand_validate], # |
                },)

            if not self.test == []: 
                # GP model is not used for testing user-requested candidates
                next_candidate = self.test.pop()
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                t1['n-m'] = np.nan
                t1['n-s'] = np.nan
                self.cand_phase = 10
                self.cand_train = False
                self.cand_validate = False
            else:
                if self.cand_train:
                    t_X_Y = self.process_training_raw(t1)
                else:
                    t_X_Y = self.process_training_raw()

                if self.cand_validate:
                    v_X_Y = self.process_validation_raw(t1)
                else:
                    v_X_Y = self.process_validation_raw()

                if self.df_train['if-validate'].sum()>=self._nbcs:
                    self.expand_bounds(self.happening_region(v_X_Y[0]))

                self.add_to_model(*t_X_Y)
                t1['bounds-size'] = self.bounds_size

                if self.df_train['if-train'].sum() < 1:                        # Phase 1: Random candidate initialisation
                    opt_phase = "Random"
                    opt_motive = 'PM'
                    self.cand_phase = 11
                    self.cand_train = True
                    self.cand_validate = False

                elif intg.bad_sim:
                    if self.opt_type == 'online':
                        opt_phase = "Reset-base"
                        opt_motive = 'reset'
                        t1['phase'] = -1
                        t1['if-train']   = True
                        t1['if-validate']= False
                        self.cand_phase = 1
                        self.cand_train = False
                        self.cand_validate = False

                    else:
                        opt_phase = "Reset-explore"
                        opt_motive = 'EI'
                        t1['phase'] = -2
                        t1['if-train']   = True
                        t1['if-validate']= False
                        self.cand_phase = 31
                        self.cand_train = True
                        self.cand_validate = False

                elif self.df_train['if-train'].sum()<self._Ainit_lim:           # Phase 2: KG
                    opt_phase = "Initialise"
                    opt_motive = 'KG'
                    self.cand_phase = 20
                    self.cand_train = True
                    self.cand_validate = False

                elif self.df_train['if-train'].sum()<self._Binit_lim:           # Phase 3: EI
                    opt_phase = "Explore"
                    opt_motive = 'EI'
                    self.cand_phase = 30
                    self.cand_train = True
                    self.cand_validate = False

                elif self.df_train['if-train'].sum()<self._Cinit_lim:           # Phase 4: PM
                    opt_phase = "Exploit"
                    opt_motive = 'PM'
                    self.cand_phase = 40
                    self.cand_train = True
                    self.cand_validate = True

                elif self.df_train['if-train'].sum()<self._Dinit_lim:           # Phase 5: s-expansion + EI
                    if self.cand_phase == 52:
                        opt_phase = "Explore+b-expansion"
                        opt_motive = 'EI'
                        self.cand_phase = 51
                    else:
                        opt_phase = "Exploit+b-expansion"
                        opt_motive = 'PM'
                        self.cand_phase = 52
                    self.cand_train = True
                    self.cand_validate = True

                elif (self.cce_true(self.df_train)>=self._nbcs
                    and not self.cand_phase == 61):                             # Phase 6: s-expansion + w-expansion + EI
                    print(f"{self.cce_true(self.df_train)} >= {self._nbcs}")

                    opt_phase = "Explore+b-expansion+w-expansion"
                    opt_motive = 'EI'
                    self.cand_phase = 61
                    self.cand_validate = False
                    self.cand_train = True

                elif len(self.df_train.index)<self._END_lim:
                    print(f"{self.cce_true(self.df_train)} < {self._nbcs}")

                    opt_phase = "Exploit+b-expansion+w-expansion"
                    opt_motive = 'PM'
                    self.cand_phase = 62
                    self.cand_train = False
                    self.cand_validate = True
                
                    intg._skip_first_n += intg._increment
                    intg._capture_next_n += intg._increment*2

                else:
                    opt_phase = "Use-only-model-optimum"
                    opt_motive = 'PM'
                    self.cand_phase = 100
                    self.cand_train = False
                    self.cand_validate = False
                
                    # Back to normal, with no more rewinding
                    intg._skip_first_n   = intg._increment*2
                    intg._capture_next_n = intg._increment*4
                    
                    intg._no_more_optimisation = True
                    intg.opt_tend = intg.tcurr
                    
                    if self.force_abort:
                        intg.abort = True

                next_candidate, t1['n-m'], t1['n-s'] = self.next_from_model(opt_motive)
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                print(f"{self.cand_phase}-{opt_motive}-{opt_phase} \t : \t {next_candidate}")

                if not opt_motive == 'PM':
                    best_candidate, t1['b-m'], t1['b-s'] = self.next_from_model('PM')
                    for i, val in enumerate(best_candidate):
                        t1[f'b-{i}'] = val
                else:
                    # next candidate is the actual best candidate
                    best_candidate = next_candidate
                    t1['b-m'], t1['b-s'] = t1['n-m'], t1['n-s']
                    for i, val in enumerate(best_candidate):
                        t1[f'b-{i}'] = val
                
            t1['opt-time'] = perf_counter() - opt_time_start

            if self.opt_type == 'online': 
                t1[self.index_name] = tcurr                
            else:
                t1[self.index_name] = len(self.df_train.index)+1

            t1['cumm-compute-time'] = intg.pseudointegrator._compute_time
            t1['capture-window'] = intg.actually_captured
            # ------------------------------------------------------------------
            # Add all the data collected into the main dataframe

            self.df_train = pd.concat([self.df_train, t1], ignore_index=True)
            # Ensure no all-NA columns in self.df_train and t1 before concat
            # self.df_train = pd.concat([self.df_train.dropna(how='all', axis=1), t1.dropna(how='all', axis=1)], ignore_index=True)

            # View results as csv file
            self.df_train.to_csv(self.outf, index=False)
            # Notify intg of the latest generated candidate and other info
            intg.candidate = self._postprocess_ccandidate(list(t1[self._n_cols].values)[0])
            # ------------------------------------------------------------------

        intg._no_more_optimisation = self.comm.bcast(intg._no_more_optimisation, root = self.root)

        if not intg._no_more_optimisation:
            intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def process_training_raw(self, t1 = None):

        new_df_train = self.df_train[self.df_train['if-train'] == True] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if t1 is not None:

            # Process all optimisables with t1 too

            tX = pd.concat([new_df_train[self._t_cand], t1[self._t_cand]], **args).astype(np.float64).to_numpy()
            tY = pd.concat([new_df_train['t-m'], t1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)

            # Ensure no all-NA columns in new_df_train[self._t_cand] and t1[self._t_cand] before concat
            # filtered_new_df_train = new_df_train[self._t_cand].dropna(how='all', axis=1)
            # filtered_t1 = t1[self._t_cand].dropna(how='all', axis=1)
            # tX = pd.concat([filtered_new_df_train, filtered_t1], **args).astype(np.float64).to_numpy()

            # If handling empty entries explicitly before concat
            # tY = pd.concat([new_df_train['t-m'].dropna(), t1['t-m'].iloc[:1].dropna()], **args).to_numpy().reshape(-1, 1)

        else:
            tX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            tY = new_df_train['t-m'].to_numpy().reshape(-1, 1)

        if len(tY) > 0:
            tY[np.isnan(tY)] = self.bad_sim_multiplier*np.nanmax(tY)

        return tX, tY

    def process_validation_raw(self, v1 = None):

        new_df_train = self.df_train[self.df_train['if-validate'] == True] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if v1 is not None:
            # Process all optimisables with t1 too
            vX = pd.concat([new_df_train[self._t_cand], v1[self._t_cand]], **args).astype(np.float64).to_numpy()
            vY = pd.concat([new_df_train['t-m'], v1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)

        else:
            vX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            vY = new_df_train['t-m'].to_numpy().reshape(-1, 1)

        if len(vY) > 0:
            # If NaN, then replace with twice the data of worst working candidate
            vY[np.isnan(vY)] = self.bad_sim_multiplier*np.nanmax(vY)

        return vX, vY

    def add_to_model(self, tX, tY):

        from gpytorch.mlls             import ExactMarginalLogLikelihood
        from botorch.models            import SingleTaskGP
        from botorch.fit               import fit_gpytorch_model as fit_model
        from botorch.models.transforms import Standardize, Normalize

        self.normalise = Normalize(d=tX.shape[1], bounds=self._bnds)
        self.standardise = Standardize(m=1)

        self.normalise.train(True)
        self.standardise.train(True)
        
        self._norm_X = self.normalise.transform(self.torch.tensor(tX , **self.torch_kwargs))
        self._stan_Y, _ = self.standardise.forward(self.torch.tensor(tY , **self.torch_kwargs))

        self.model = SingleTaskGP(train_X = self._norm_X, 
                                  train_Y = self._stan_Y,)

        mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, 
                                         model = self.model)

        mll = mll.to(**self.torch_kwargs)                      
        fit_model(mll)

        self.normalise.train(False)
        self.standardise.train(False)

    @property
    def bounds(self):
        return self._bnds.detach().cpu().numpy()

    @bounds.setter
    def bounds(self, y):
        self._bnds = self.torch.tensor(y, **self.torch_kwargs)

    def happening_region(self, tX):
        
        best_cands = tX # tX[-self._nbcs:]      # A big change to bounds-expansion

        if len(best_cands) <= 2 :
            return

        means, stds = np.mean(best_cands, axis=0), np.std( best_cands, axis=0)

        hr = self.torch.tensor(np.array([(1.0-self.mean_mult)*means - self.std_mult * stds,
                                         (1.0+self.mean_mult)*means + self.std_mult * stds]), 
                               **self.torch_kwargs)
        
        print(f"{means = }, {stds = }, \n Happening-region \n {hr}  \n previous bounds \n {self._bnds} ")

        return hr

    def expand_bounds(self, hr):
        if hr is None:
            return

        l_bnds_inc_loc = self._bbnds[0, :] > hr[0, :]
        u_bnds_inc_loc = self._bbnds[1, :] < hr[1, :]
        
        self._bnds[0, l_bnds_inc_loc] = hr[0, l_bnds_inc_loc]
        self._bnds[1, u_bnds_inc_loc] = hr[1, u_bnds_inc_loc]

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

        if any(opt.startswith(         'cstep:') for opt in self.optimisables):  n_csteps = sum(opt.startswith(  'cstep:') for opt in self.optimisables);  csteps = self._preprocess_csteps(pseudointegrator.csteps , n_csteps )
        if any(opt.startswith(       'cstep-a:') for opt in self.optimisables): n_cstepsa = sum(opt.startswith('cstep-a:') for opt in self.optimisables); cstepsa = self._preprocess_csteps(pseudointegrator.cstepsa, n_cstepsa)
        if any(opt.startswith(       'cstep-b:') for opt in self.optimisables): n_cstepsb = sum(opt.startswith('cstep-b:') for opt in self.optimisables); cstepsb = self._preprocess_csteps(pseudointegrator.cstepsb, n_cstepsb)
        if any(opt.startswith(       'cstep-c:') for opt in self.optimisables): n_cstepsc = sum(opt.startswith('cstep-c:') for opt in self.optimisables); cstepsc = self._preprocess_csteps(pseudointegrator.cstepsc, n_cstepsc)
        if any(opt.startswith('pseudo-dt-fact:') for opt in self.optimisables):    dtaufs = pseudointegrator.dtaufs
        if any(opt.startswith( 'pseudo-dt-max:') for opt in self.optimisables): dtau_maxs = pseudointegrator.dtau_maxs

        # Check consistency of the number of optimisables
        if any(opt.startswith('pseudo-dt-fact:') for opt in self.optimisables) and len(dtaufs)    != self.depth+1: raise ValueError(f"{len(dtaufs   ) = } neq 1+{self.depth}")
        if any(opt.startswith( 'pseudo-dt-max:') for opt in self.optimisables) and len(dtau_maxs) != self.depth+1: raise ValueError(f"{len(dtau_maxs) = } neq 1+{self.depth}")

        for opt in self.optimisables:
            # Array optimisables, those with colon
            if   opt.startswith(         'cstep:') and opt[ 6:].isdigit(): unprocessed.append(   csteps[int(opt[ 6:])])
            elif opt.startswith(       'cstep-a:') and opt[ 8:].isdigit(): unprocessed.append(  cstepsa[int(opt[ 8:])])
            elif opt.startswith(       'cstep-b:') and opt[ 8:].isdigit(): unprocessed.append(  cstepsb[int(opt[ 8:])])
            elif opt.startswith(       'cstep-c:') and opt[ 8:].isdigit(): unprocessed.append(  cstepsc[int(opt[ 8:])])
            elif opt.startswith( 'pseudo-dt-max:') and opt[14:].isdigit(): unprocessed.append(dtau_maxs[int(opt[14:])])
            elif opt.startswith('pseudo-dt-fact:') and opt[15:].isdigit(): unprocessed.append(   dtaufs[int(opt[15:])])
            # Non-array optimisables
            elif opt == 'pseudo-dt-max':                           unprocessed.append(pseudointegrator.pintg.Δτᴹ)
            elif opt == 'pseudo-dt-fact':                          unprocessed.append(pseudointegrator.dtauf)
            else:
                raise ValueError(f"Unrecognised optimisable: {opt}")

        return unprocessed

    def _postprocess_ccandidate(self, ccandidate):
        post_processed = {}
        for i, opt in enumerate(self.optimisables):
            # Array optimisables, those with colon
            if   opt.startswith(         'cstep:') and opt[ 6:].isdigit(): post_processed[opt] = ccandidate[i]
            elif opt.startswith(       'cstep-a:') and opt[ 8:].isdigit(): post_processed[opt] = ccandidate[i]
            elif opt.startswith(       'cstep-b:') and opt[ 8:].isdigit(): post_processed[opt] = ccandidate[i]
            elif opt.startswith(       'cstep-c:') and opt[ 8:].isdigit(): post_processed[opt] = ccandidate[i]
            elif opt.startswith('pseudo-dt-fact:') and opt[15:].isdigit(): post_processed[opt] = ccandidate[i]
            elif opt.startswith( 'pseudo-dt-max:') and opt[14:].isdigit(): post_processed[opt] = ccandidate[i]
            # Non-array optimisables
            elif opt ==  'pseudo-dt-max': post_processed[opt] = ccandidate[i]
            elif opt == 'pseudo-dt-fact': post_processed[opt] = ccandidate[i]
            else:
                raise ValueError(f"Unrecognised optimisable {opt}")
        return post_processed

    def _preprocess_csteps(self, csteps, n_csteps):
        if   n_csteps == 5: return csteps[0], csteps[1], csteps[self.depth], csteps[-2], csteps[-1],
        elif n_csteps == 4: return            csteps[1], csteps[self.depth], csteps[-2], csteps[-1],
        elif n_csteps == 3: return            csteps[1], csteps[self.depth],             csteps[-1],
        elif n_csteps == 2: return                       csteps[self.depth],             csteps[-1],
        elif n_csteps == 1: return                                                       csteps[-1],
        else:
            raise ValueError(f"Unrecognised number of csteps {n_csteps}")

    def next_from_model(self, type):

        from botorch.acquisition           import qKnowledgeGradient
        from botorch.acquisition.objective import ScalarizedPosteriorTransform

        from botorch.acquisition           import qNoisyExpectedImprovement
        from botorch.acquisition.objective import ScalarizedPosteriorTransform

        from botorch.acquisition import PosteriorMean

        if type == 'KG':
            samples   = 1024
            restarts  =   10
            fantasies =   64 # 128

            _acquisition_function = qKnowledgeGradient(
                self.model, 
                posterior_transform = ScalarizedPosteriorTransform(weights=self.negw),
                num_fantasies       = fantasies,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'EI':
            samples   = 1024
            restarts  =   10

            _acquisition_function = qNoisyExpectedImprovement(
                self.model, self._norm_X,
                posterior_transform = ScalarizedPosteriorTransform(weights=self.negw),
                prune_baseline = True,
                )
            X_cand = self.optimise(_acquisition_function, samples, restarts)
            X_b = self.normalise.untransform(X_cand)

        elif type == 'PM':
            _acquisition_function = PosteriorMean(self.model, maximize = False)

            raw_samples  = 2048 
            num_restarts =   20 

            X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
            X_b    = self.normalise.untransform(X_cand)

        elif type == 'reset':
            # Get the first row candidates and convert them to a tensor
            X_b1 = self.df_train[self._t_cand].iloc[0].astype(np.float64).to_numpy()
            X_b = self.torch.tensor(X_b1, **self.torch_kwargs)

        else:
            raise ValueError(f'next_type {type} not recognised')

        return self.substitute_in_model(X_b)

    def optimise(self, _acquisition_function, raw_samples, num_restarts):
        from botorch.optim.optimize import optimize_acqf
        X_cand, _ = optimize_acqf(acq_function = _acquisition_function, q = 1,
            bounds = self.normalise.transform(self._bnds),
            num_restarts = num_restarts,raw_samples = raw_samples,
            )
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
        Y_sem = (Y_u - Y_l)/4

        XX = X_sub.detach().cpu().squeeze().tolist()

        if isinstance(XX, float):
            return [XX], Y_m, Y_sem
        else:
            return XX, Y_m, Y_sem

    def columns_from_optimisables(self):

        self.columns = [
            *[f't-{i}' for i in range(len(self.optimisables))],'t-m', 't-e', 't-d', 
            *[f'n-{i}' for i in range(len(self.optimisables))],'n-m', 'n-s', 
            *[f'b-{i}' for i in range(len(self.optimisables))],'b-m', 'b-s',
            'opt-time', 'cumm-compute-time', 
            'if-train', 'if-validate', 'capture-window', 
            'bounds-size', 
            'phase'
            ] 

        self._t_cols = list(filter(lambda x: x.startswith('t-'), self.columns))
        self._t_cand = list(filter(lambda x: x.startswith('t-') and x[2:].isdigit(), self.columns))
        
        self._b_cols = list(filter(lambda x: x.startswith('b-'), self.columns))
        self._b_cand = list(filter(lambda x: x.startswith('b-') and x[2:].isdigit(), self.columns))
        
        self._n_cols = list(filter(lambda x: x.startswith('n-'), self.columns))
        self._n_cand = list(filter(lambda x: x.startswith('n-') and x[2:].isdigit(), self.columns))

    def cce_true(self, df: pd.DataFrame) -> int:
        continuous_true_count = 1

        for value in df['if-validate'].values[::-1]:
            if value:
                continuous_true_count += 1
            else:
                break

        return continuous_true_count
    