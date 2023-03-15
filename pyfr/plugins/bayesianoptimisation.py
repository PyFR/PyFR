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

        self.optimisables = self.cfg.getliteral(cfgsect, 'optimisables')
        self.bad_sim_multiplier = self.cfg.getfloat(cfgsect, 'bad-sim-multiplier', 1.5)
        self.columns_from_optimisables()

        self._nbcs       = len(self.optimisables) # Number of best candidates to consider
        self._A_lim      = self.cfg.getfloat(cfgsect, 'A-lim', 0.75*2**self._nbcs) # when to start looking                                                                  (KG)     48
        self._B_lim      = self.cfg.getfloat(cfgsect, 'B-lim', 1.00*2**self._nbcs)                                                     # When start checking model quality  (EI)     64
        self._C_lim      = self.cfg.getfloat(cfgsect, 'C-lim', 1.25*2**self._nbcs)                                                     # When start looking for optimum     (KG+PM)  80
        self._D_lim      = self.cfg.getfloat(cfgsect, 'D-lim', 1.50*2**self._nbcs)                                                     # When start checking optimum        (EI+PM)  96
        self._E_lim      = self.cfg.getfloat(cfgsect, 'E-lim', 2   *2**self._nbcs)                                                     # When start checking optimum        (EI+PM) 128
        self._END_lim    = self.cfg.getfloat(cfgsect, 'stop-offline-simulation', 4*self._A_lim) # Buffer kCV and LooCV and delete more       256
        self.force_abort = self.cfg.getbool( cfgsect, 'force-abort', False)

        intg.opt_type = self.opt_type = suffix
        if suffix == 'online':
            self.columns.append('tcurr')
            self.index_name = 'tcurr'
        elif suffix == 'onfline':
            self.columns.append('iteration')
            self.index_name = 'iteration'
            intg.offline_optimisation_complete = False
        elif suffix == 'offline':
            raise NotImplementedError(f'offline not implemented.')
        else:
            raise ValueError('Invalid suffix')

        self.LooCV_limit = self.cfg.getfloat(cfgsect, 'loocv-limit', 1.0)
        self.kCV_limit = self.cfg.getfloat(cfgsect, 'kcv-limit', 0.5)

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
            self.test = self.cfg.getliteral(cfgsect, 'test',[])

            # The latest tested candidate characteristics are determined by t1 dataframe
            # The following candidates determine what the next candidate should be like
            self.cand_phase = 1        # First canddiate
            self.cand_train = True     # First canddiate
            self.cand_validate = False # First canddiate

        self.deserialise(intg, data)

    def __call__(self, intg):

        if not intg.reset_opt_stats:
        # If optimisation statistics collected by the previous plugin is not reset yet
        # then don't apply Bayesian Optimisation to the data
            return

        # self.check_offline_optimisation_status(intg)

        if self.rank == self.root:
            opt_time_start = perf_counter()

            if self.opt_type == 'online' and intg.bad_sim:
                if np.isnan(intg.opt_cost_mean) and self.df_train.empty:
                    raise ValueError("Initial configuration must be working.")
                tcurr = self.df_train.iloc[-1, self.df_train.columns.get_loc('tcurr')]
            else:
                tcurr = intg.tcurr

            if self.df_train.empty:
                # Set the reference acceptable error in cost
                print(f"Setting error from {intg._stability} to {3*intg.opt_cost_sem}")
                intg._stability = 2*intg.opt_cost_sem

                # Since an imprecise candidate simulation will be rewound, be careful
                # Start with a conservative 2X
                # A good candidate MUST NOT BE KILLED THIS WAY
                intg._precision = 5*intg.opt_cost_std

            # Convert last iteration data from intg to dataframe
            tested_candidate = self.candidate_from_intg(intg.pseudointegrator)

            t1 =  pd.DataFrame({
                **{f't-{i}': [val] for i, val in enumerate(tested_candidate)},
                't-m': [intg.opt_cost_mean], 
                't-d': [intg.opt_cost_std], 
                't-e': [intg.opt_cost_sem], 
                'phase':[self.cand_phase],           # |   
                'if-train': [self.cand_train],       # |----> Latest tested candidate 
                'if-validate': [self.cand_validate], # |
                },)

            if not self.test == []: 
                # GP model is not used for this
                next_candidate = self.test.pop()
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                t1['n-m'] = np.nan
                t1['n-s'] = np.nan
                self.cand_phase = 10
                self.cand_train = False
                self.cand_validate = False
            else:

                # We need to add the latest candidate to the model only if 
                #   it is for training (not validation)
                # Else we add it to the validation set
                if self.cand_train and not self.cand_validate:
                    t_X_Y_Yv = self.process_training_raw(t1)
                    v_X_Y_Yv = self.process_validation_raw()
                elif not self.cand_train and self.cand_validate:
                    v_X_Y_Yv = self.process_validation_raw(t1)
                    t_X_Y_Yv = self.process_training_raw()
                else:
                    v_X_Y_Yv = self.process_validation_raw()
                    t_X_Y_Yv = self.process_training_raw()

                # ------------------------------------------------------------------
                # NOVEL IDEA: INCREASE BOUNDS ONLY AFTER EXPLORATION AND ONLY WITH VALIDATION DATA
                # ------------------------------------------------------------------
                # NEW AREAS SHOULD BE EXPLORED ONCE EXISTING AREAS ARE MODELLED WELL
                # KEEP STRESSING THE OPTIMISER TO FIND NEW AREAS TO EXPLORE
                # VALIDATION DATA EXPLORES WELL BECAUSE IT IS AROUND THE OPTIMUM
                # ------------------------------------------------------------------

                # Increase bounds only after exploration
                if self.df_train['if-validate'].sum()>=self._nbcs:
                    # Set bounds on the basis of training data 
                    self.expand_bounds(self.happening_region(intg, v_X_Y_Yv[0]))

                self.add_to_model(*t_X_Y_Yv)

                # Get loocv error with the validation set and training-set
                t1['bounds-size'] = self.bounds_size
                t1['LooCV'] = loocv_err = self.loocv_error
                loocv_err = 0 if loocv_err==None else loocv_err
                t1['KCV'] = kcv_err = self.kcv_error(*v_X_Y_Yv)
                kcv_err = 0 if kcv_err==None else kcv_err


                # ------------------------------------------------------------------
                if intg.bad_sim and self.df_train['phase'].iloc[-1]==0 and self.opt_type == 'online':
                # We shall assume that the first candidate is a good candidate
                #   This is because the first candidate is the initial configuration
                #   and all our tolerances are set on the basis of this candidate
                # If we use our first candidate to reset online simulation and  
                #       takes more time to converge ... then its fine
                #       starts to NaN               ... then we have a problem
                    intg.bad_sim = False
                    if intg.opt_cost_std < intg._precision:
                        raise ValueError("Something's wrong here.")                        
                    else:
                        print("Base simulation was marked as bad due to "
                              "high deviation in cost of the first candidate."
                              "This is fine.")

                # Check if optimisation is performing alright
                if self.df_train.empty:
                    # Initialisation phase
                    opt_motive = 'PM'
                    self.cand_phase = 11
                    self.cand_train = True
                    self.cand_validate = False

                # Check if the last candidate was a the base candidate
                elif intg.bad_sim:
                    # Fall-back
                    print("Bad simulation.")

                    if self.opt_type == 'online':
                        # We will reset the solution fields with something we are confident of
                        # This wil help us over-ride the source of bad candidate effects
                        # We may not know the source, (which among the last 2 candidates caused this) and still fix it 
                        
                        opt_motive = 'reset'
                        t1['phase'] = -1
                        t1['if-train']   = True
                        t1['if-validate']= False
                        self.cand_phase = 1
                        self.cand_train = False
                        self.cand_validate = False

                    # Else its definitely 'onfline'
                    elif self.cand_validate:
                        # Make sure you still continue to build up on both the model and the validation set alternatively

                        opt_motive = 'EI'
                        t1['phase'] = -31
                        t1['if-train']   = True
                        t1['if-validate']= False
                        self.cand_phase = 31
                        self.cand_train = True
                        self.cand_validate = False

                    else:
                        opt_motive = 'PM'
                        t1['phase'] = -32
                        t1['if-train']   = True
                        t1['if-validate']= False
                        self.cand_phase = 32
                        self.cand_train = False
                        self.cand_validate = True

                # ------------------------------------------------------------------
                # NOVEL IDEA: MULTI-STEP OPTIMISATION STRATEGY
                # ------------------------------------------------------------------
                # FIRST STEP: INITIALISATION WITH KG
                # SECOND STEP: EXPLORATION WITH KG AND PARALLEL VALIDATION WITH PM
                # THIRD STEP: EXPLOITATION WITH EI AND PARALLEL VALIDATION WITH PM
                # FOURTH STEP: STRESS-TESTING FINAL MODEL WITH PM
                # ------------------------------------------------------------------

                elif self.df_train['if-train'].sum()<self._A_lim:       # 48
                    # Initialisation phase - I
                    print("Initialisation training with KG.")
                    opt_motive = 'KG'
                    self.cand_phase = 20
                    self.cand_train = True
                    self.cand_validate = False

                elif self.df_train['if-train'].sum()<self._B_lim:      # 64
                    # Initialisation phase - II
                    print("Quick-optimum-search training with EI.")
                    opt_motive = 'EI'
                    self.cand_phase = 30
                    self.cand_train = True
                    self.cand_validate = False

                elif loocv_err>self.LooCV_limit:
                    # Initialisation phase - II
                    if self.cand_validate:
                        print("Exploration training with KG.")
                        opt_motive = 'KG'
                        self.cand_phase = 21
                        self.cand_train = True
                        self.cand_validate = False
                    else:
                        print("Exploration validation with PM.")
                        opt_motive = 'PM'
                        self.cand_phase = 22
                        self.cand_train = False
                        self.cand_validate = True

                elif kcv_err>self.kCV_limit:
                    if self.cand_validate:
                        print("Exploitative EI phase.")
                        opt_motive = 'EI'
                        self.cand_phase = 33
                        self.cand_train = True
                        self.cand_validate = False
                    else:
                        print("Exploitative PM phase.")
                        opt_motive = 'PM'
                        self.cand_phase = 34
                        self.cand_train = False
                        self.cand_validate = True

                else:
                    # Finalising phase
                    print("Finalising phase.")
                    opt_motive = 'PM'
                    self.cand_phase = 41
                    self.cand_train = False
                    self.cand_validate = True

                next_candidate, t1['n-m'], t1['n-s'] = self.next_from_model(opt_motive)
                for i, val in enumerate(next_candidate):
                    t1[f'n-{i}'] = val
                print(f"{opt_motive}: {next_candidate}")

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
                
                # If only to get data for paper, then tend is meaningless
                if (self.force_abort
                    and (len(self.df_train.index)>self._END_lim
                        or intg.tcurr>intg.tend)):
                    intg.abort = True
                    
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
            # ------------------------------------------------------------------
            
            if self.df_train['LooCV'].count() > self._nbcs:
                # Get a rolling mean of self.df_train['LooCV','KCV']
                self.df_train[f'roll{self._nbcs}-diff-LooCV'] = self.df_train['LooCV'].rolling(window=self._nbcs).mean().diff()

                if (self.df_train[f'roll{self._nbcs}-diff-LooCV'].iloc[-1] > 0 
                    and self.df_train['if-train'].sum() > self._C_lim):              # Figure out the critical number of datapooints for us to apply this to offline and online both
                                                                 # If simulation time is more than 10% of first simulation time then number of datapoints should be noted
                    # Find the index of the first occurance of self.df_train['if-train'] == True 
                    position = self.df_train[self.df_train['if-train']].index[0]
                    self.df_train.loc[position, 'if-train'] = False

                    # ------------------------------------------------------------------
                    # NOVEL IDEA: IF UNABLE TO MODEL WELL, INCREASE CAPTURE TIME FOR TRAINING
                    # ------------------------------------------------------------------
                    # HYPERTROPHY/SUCCESSIVE PROGRESSIVE LOADING
                    # KEEP STRESSING THE OPTIMISER AT ITS BAD TIMES SO ITS ADAPTIVE ENOUGH
                    # ------------------------------------------------------------------
                    if self.cand_train and not self.cand_validate:
                        intg._skip_first_n      += intg._increment
                        intg._capture_next_n    += intg._increment*2
                        intg._stabilise_final_n += intg._increment*2
                    # ------------------------------------------------------------------

            # If intg.actually_captured is equal to or greater than + intg._capture_next_n + intg._stabilise_final_n then 
            if (intg.actually_captured >= intg._capture_next_n + intg._increment
                ) or (self.df_train['if-train'].sum() > self._E_lim
                ) or (self.df_train['phase'].iloc[-1] < 0 and self.df_train['phase'].iloc[-2] < 0   
                      ):  # If last 2 values in self.df_train['phase'] are negative then

                position = self.df_train[self.df_train['if-train']].index[0]
                self.df_train.loc[position, 'if-train'] = False

                intg._skip_first_n      += intg._increment*2
                intg._capture_next_n    += intg._increment*4
                intg._stabilise_final_n += intg._increment*4

            # ------------------------------------------------------------------
            # View results as csv file
            self.df_train.to_csv(self.outf, index=False)
            # ------------------------------------------------------------------
            # Notify intg of the latest generated candidate and other info
            intg.candidate = self._postprocess_ccandidate(list(t1[self._n_cols].values)[0])
            # ------------------------------------------------------------------

        intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def serialise(self, intg):
        if self.rank == self.root:
            return {'df_train':self.df_train.to_numpy(dtype=np.float64),
                    }

    def deserialise(self, intg, data):

        intg.candidate = {}
        self.depth = self.cfg.getint('solver', 'order', 0)

        if self.rank == self.root:

            if bool(data):
                self.df_train = pd.DataFrame(data['df_train'], columns=self.columns)

                # Set the candidate to the next candidate if optimisation was being performed in the previous run                
                next_candidate = self.df_train[self._n_cols].tail(1).to_numpy(dtype=np.float64)[0]
                self.cand_train = self.df_train['if-train'].tail(1).to_numpy(dtype=np.bool)[0]
                self.cand_validate = self.df_train['if-validate'].tail(1).to_numpy(dtype=np.bool)[0]

                intg._skip_first_n   = int(0.5*self.df_train['capture-window'].tail(1).to_numpy(dtype=np.float64)[0])
                intg._capture_last_n = self.df_train['capture-window'].tail(1).to_numpy(dtype=np.float64)[0]
                intg.candidate = self._postprocess_ccandidate(next_candidate)
                
            else:
                self.df_train = pd.DataFrame(columns=self.columns)
            
        if (self.rank == self.root) and len(intg.candidate)>0:
            intg.candidate = self.comm.bcast(intg.candidate, root = self.root)

    def process_training_raw(self, t1 = None):

        new_df_train = self.df_train[self.df_train['if-train'] == True] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if t1 is not None:
            # Process all optimisables with t1 too
            tX = pd.concat([new_df_train[self._t_cand], t1[self._t_cand]], **args).astype(np.float64).to_numpy()
            tY = pd.concat([new_df_train['t-m'], t1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)

            if self.opt_type == 'online':
                tYv = pd.concat([new_df_train['t-d'], t1['t-d'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2
            else:
                tYv = pd.concat([new_df_train['t-e'], t1['t-e'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2

        else:
            tX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            tY = new_df_train['t-m'].to_numpy().reshape(-1, 1)

            if self.opt_type == 'online':
                tYv = new_df_train['t-d'].to_numpy().reshape(-1, 1) ** 2
            else:
                tYv = new_df_train['t-e'].to_numpy().reshape(-1, 1) ** 2

        if len(tY) > 0:
            # If NaN, then replace with twice the data of worst working candidate
            tY[np.isnan(tY)] = self.bad_sim_multiplier*np.nanmax(tY)
            tYv[np.isnan(tYv)] = self.bad_sim_multiplier*np.nanmax(tYv)
            tYv = tYv.reshape(-1,1)

        return tX, tY, tYv

    def process_validation_raw(self, v1 = None):

        new_df_train = self.df_train[self.df_train['if-validate'] == True] 
        args = {'axis' : 0, 'ignore_index' : True, 'sort' : False}

        if v1 is not None:
            # Process all optimisables with t1 too
            vX = pd.concat([new_df_train[self._t_cand], v1[self._t_cand]], **args).astype(np.float64).to_numpy()
            vY = pd.concat([new_df_train['t-m'], v1['t-m'].iloc[:1]], **args).to_numpy().reshape(-1, 1)

            if self.opt_type == 'online':
                vYv = pd.concat([new_df_train['t-d'], v1['t-d'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2
            else:
                vYv = pd.concat([new_df_train['t-e'], v1['t-e'].iloc[:1]], **args).to_numpy().reshape(-1, 1) ** 2

        else:
            vX = new_df_train[self._t_cand].astype(np.float64).to_numpy()
            vY = new_df_train['t-m'].to_numpy().reshape(-1, 1)

            if self.opt_type == 'online':
                vYv = new_df_train['t-d'].to_numpy().reshape(-1, 1) ** 2
            else:
                vYv = new_df_train['t-e'].to_numpy().reshape(-1, 1) ** 2

        if len(vY) > 0:
            # If NaN, then replace with twice the data of worst working candidate
            vY[np.isnan(vY)] = self.bad_sim_multiplier*np.nanmax(vY)
            vYv[np.isnan(vYv)] = self.bad_sim_multiplier*np.nanmax(vYv)
            vYv = vYv.reshape(-1,1)

        return vX, vY, vYv

    def add_to_model(self, tX, tY, tYv):
        """ Fit a Fixed Noise Gaussian Process model on given data

        Args:
            tX (numpy.ndarray): Training input
            tY (numpy.ndarray): Training output
            tYv (numpy.ndarray): Training output variance
        """

        from gpytorch.mlls             import ExactMarginalLogLikelihood
        from botorch.models            import FixedNoiseGP, SingleTaskGP
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

        if self.opt_type == 'online':
            self.model = SingleTaskGP(train_X = self._norm_X, 
                                      train_Y = self._stan_Y,)
        else:
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
        if self.df_train['if-train'].sum()>=self._A_lim:
                                                                                # Calculate LOOCV error only after the initialising 16 iterations
                                                                                # This is to avoid wasting time in the start
                                                                                #   when model is definitely not good enough
            return self.cv_folds(Train_X    = self._norm_X, 
                                Train_Y    = self._stan_Y, 
                                Train_Yvar = self._stan_Yvar,
                                ).detach().cpu().numpy()
        else:
            return None
            
    def kcv_error(self, vX, vY, vYv):
        # Based on processing done like normalise and standardise, process vX

        if len(vY) == 0:
            return None

        _norm_vX = self.normalise.transform(
            self.torch.tensor(vX[-self._nbcs:] , **self.torch_kwargs)) 
        _stan_vY, _stan_vYv = self.standardise.forward(
            self.torch.tensor(vY[-self._nbcs:] , **self.torch_kwargs),
            self.torch.tensor(vYv[-self._nbcs:], **self.torch_kwargs))

        if self.df_train['if-train'].sum()>=1:
            # Calculate LOOCV error only after the initialising 16 iterations
            # This is to avoid wasting time in the start
            #   when model is definitely not good enough

            return self.cv_folds(Train_X = self._norm_X, Train_Y = self._stan_Y, Train_Yvar = self._stan_Yvar,
                                Val_X = _norm_vX, Val_Y = _stan_vY, Val_Yvar = _stan_vYv,
                                ).detach().cpu().numpy()

        else:
            return None
            
    @property
    def bounds(self):
        return self._bnds.detach().cpu().numpy()

    @bounds.setter
    def bounds(self, y):
        self._bnds = self.torch.tensor(y, **self.torch_kwargs)

    def happening_region(self, intg, tX):
        """ Expand the bounds based on the last 16 PM points
            How?
            1. Find the mean and standard deviation of the last 16 points
            2. Find the happening region as 1+-10% mean +- 2*std in each direction
            3. Take union of the happening region and the initial bounds
        """
        
        mean_var = 0.5 # NEXT TEST: 0.5                      # Extra wiggle-room for hr around mean
        std_mult = 2  # NEXT TEST: 5                        # If wiggling too much, search more around here

        if self.opt_type == 'online':
            mean_var = 0.5 # NEXT TEST: 0.5                      # Extra wiggle-room for hr around mean
            std_mult = 2  # NEXT TEST: 5                        # If wiggling too much, search more around here
        else:
            mean_var = 0.5
            std_mult = 5

        best_cands = tX[-self._nbcs:]

        if len(best_cands) <= 2 :
            return

        means, stds = np.mean(best_cands, axis=0), np.std( best_cands, axis=0)

        # BEFORE
        # hr = self.torch.tensor(np.array([means - std_mult * stds,
        #                                 (means + std_mult * stds]), 
        #                       **self.torch_kwargs)

        # AFTER        
        hr = self.torch.tensor(np.array([(1.0-mean_var)*means - std_mult * stds,
                                         (1.0+mean_var)*means + std_mult * stds]), 
                               **self.torch_kwargs)
        
        print(f"{means = }, {stds = }, \n Happening-region \n {hr}  \n previous bounds \n {self._bnds} ")

        return hr

    def expand_bounds(self, hr):
        if hr is None:
            return

        # Increase bounds to include happening region
        l_bnds_inc_loc = self._bbnds[0, :] > hr[0, :]
        u_bnds_inc_loc = self._bbnds[1, :] < hr[1, :]
        
        self._bnds[0, l_bnds_inc_loc] = hr[0, l_bnds_inc_loc]
        self._bnds[1, u_bnds_inc_loc] = hr[1, u_bnds_inc_loc]

        # REMOVED UBon!!!!!
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

        if n_csteps == 5:
            return csteps[0], csteps[1], csteps[self.depth], csteps[-2], csteps[-1]
        if n_csteps == 4:
            return csteps[1], csteps[self.depth], csteps[-2], csteps[-1]
        elif n_csteps == 3:
            return csteps[1], csteps[self.depth], csteps[-1]
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
            fantasies =   64

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
        Y_sem = (Y_u - Y_l)/4

        XX = X_sub.detach().cpu().squeeze().tolist()

        if isinstance(XX, float):
            return [XX], Y_m, Y_sem
        else:
            return XX, Y_m, Y_sem
        
    def cv_folds(self, Train_X, Train_Y, Train_Yvar = None, 
                 Val_X = None, Val_Y=None, Val_Yvar=None):

        from botorch.cross_validation import CVFolds, gen_loo_cv_folds
        from botorch.cross_validation import batch_cross_validation
        from botorch.models import FixedNoiseGP, SingleTaskGP
        from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        if Val_X is None or Val_Y is None:
            # LooCV with inferred noise
            cv_folds = gen_loo_cv_folds(
                train_X    = Train_X, 
                train_Y    = Train_Y, 
                train_Yvar = Train_Yvar)
        else:
            # kCV with inferred noise
            cv_folds = CVFolds(
                train_X   = Train_X   , test_X    = Val_X, 
                train_Y   = Train_Y   , test_Y    = Val_Y, 
                train_Yvar= Train_Yvar, test_Yvar = Val_Yvar)

        # instantiate and fit model
        cv_results = batch_cross_validation(model_cls=FixedNoiseGP,
            mll_cls=ExactMarginalLogLikelihood, cv_folds=cv_folds,)

        posterior = cv_results.posterior
        mean = posterior.mean
        
        # Mean Squared Error
        cv_error = ((cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean().sqrt()
        
        return cv_error

    def columns_from_optimisables(self):

        self.columns = [
            *[f't-{i}' for i in range(len(self.optimisables))],'t-m', 't-e', 't-d', 
            *[f'n-{i}' for i in range(len(self.optimisables))],'n-m', 'n-s', 
            *[f'b-{i}' for i in range(len(self.optimisables))],'b-m', 'b-s',
            'opt-time', 'cumm-compute-time', 
            'if-train', 'if-validate', 'capture-window', 
            'LooCV',  'KCV',
            'bounds-size', 
            'phase'
            ] 

        self._t_cols = list(filter(lambda x: x.startswith('t-'), self.columns))
        self._t_cand = list(filter(lambda x: x.startswith('t-') and x[2:].isdigit(), self.columns))
        
        self._b_cols = list(filter(lambda x: x.startswith('b-'), self.columns))
        self._b_cand = list(filter(lambda x: x.startswith('b-') and x[2:].isdigit(), self.columns))
        
        self._n_cols = list(filter(lambda x: x.startswith('n-'), self.columns))
        self._n_cand = list(filter(lambda x: x.startswith('n-') and x[2:].isdigit(), self.columns))
