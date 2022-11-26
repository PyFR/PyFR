from time import perf_counter
import numpy as np
import pandas as pd

from pyfr.plugins.base import BasePlugin

class BayesianOptimisationPlugin(BasePlugin):
    name = 'bayesian_optimisation'
    systems = ['*']
    formulations = ['dual']
    
    def __init__(self, intg, cfgsect, suffix):
        
        super().__init__(intg, cfgsect, suffix)

        import torch
        self.torch = torch
        self.seed = self.cfg.getint(cfgsect, 'seed', 0)

        bounds_init = list(zip(*self.cfg.getliteral(cfgsect, 'bounds')))
        self.outf  = self.cfg.get(cfgsect, 'file' , 'bayesopt.csv')

        self.torch_kwargs = {'dtype' : torch.float32, 'device': torch.device("cpu")}    # Stress test with gpu too, compare timings

        self._bounds = self.torch.tensor(bounds_init, **self.torch_kwargs)

        self.pd_opt =  pd.DataFrame(columns = ['test-candidate', 'test-m', 'test-s',
#                                               'check-m', 'check-s',
                                               'next-candidate', 'next-m', 'next-s',
                                               'best-candidate', 'best-m', 'best-s',
                                               ])

        self.rewind_counts = 0

#        arrays = [
#                    ["test",      "test", "test", "next",      "next", "next", "best",      "best", "best", ],
#                    ["candidate", "mean", "std" , "candidate", "mean", "std" , "candidate", "mean", "std" , ],
#                    ]
#        index = pd.MultiIndex.from_arrays(arrays, names=["candidate-type", "values"])
#        print(index)

        intg.candidate = {}     # This will be used by the next plugin in line

    def __call__(self, intg):

        if not intg.reset_opt_stats:
            return

        opt_time_start = perf_counter()

        #if np.isnan(intg.opt_cost_mean) or np.isnan(intg.opt_cost_std):
#
        #    if not (np.isnan(intg.opt_cost_mean) and np.isnan(intg.opt_cost_std)):
        #        raise ValueError("Either both or none of opt_cost_mean and opt_cost_std should be NaN")
#
        #    intg.opt_cost_mean = 1.1*self.pd_opt['test-m'].max()
        #    intg.opt_cost_std  = 1.1*self.pd_opt['test-s'].max()           
#
        t1 =  pd.DataFrame({'test-candidate': [self._preprocess_csteps(intg.pseudointegrator.csteps)] , 
                            'test-m'        : [intg.opt_cost_mean], 
                            'test-s'        : [intg.opt_cost_std ]})

        # Process all optimisables
        tX  = np.array(list(self.pd_opt['test-candidate'])+[list(t1['test-candidate'])[0]])
        tY  = np.array(list(self.pd_opt['test-m'        ])+[list(t1['test-m'        ])[0]]).reshape(-1,1)
        tYv = np.array(list(self.pd_opt['test-s'        ])+[list(t1['test-s'        ])[0]])**2

        tY[ np.isnan(tY )] = np.nanmax(tY)
        tYv[np.isnan(tYv)] = np.nanmax(tYv)

        tYv = tYv.reshape(-1,1)
        self.add_to_model(tX, tY, tYv)

#         t1 = self.store_test_from_model(t1)
        #t1 = self.store_validation_from_model(t1, 0, (0, 0, 0, 0))
        #t1 = self.store_validation_from_model(t1, 1, (1, 1, 1, 1))
        #t1 = self.store_validation_from_model(t1, 5, (5, 5, 5, 5))
        
#        t1['best-candidate'], t1['best-m'], t1['best-s'] = self.best_from_model()
        t1['best-candidate'], t1['best-m'], t1['best-s'] = self.best_from_model()

        if intg.rewind:
            self.rewind_counts += 1

        if self.pd_opt.empty:
            t1['base-time'] = self.ct_first = intg.pseudointegrator._compute_time
            self.start_PM = False
            last_opt_time = 0
        else:
            t1['base-time'] = (1+self.pd_opt.count(0)[0]-self.rewind_counts)*self.ct_first
            self.start_PM = list(self.pd_opt['cumm-time'].iloc[[-1]] < self.pd_opt['base-time'].iloc[[-1]])[0]
            last_opt_time = self.pd_opt.iloc[-1, self.pd_opt.columns.get_loc('cumm-opt-time')]

        if ((0.9*self.pd_opt['test-m'].min()) > t1['best-m'].tail(1).min()) or self.start_PM or intg.rewind:

            print(f"{((0.9*self.pd_opt['test-m'].min()) > t1['best-m'].tail(1).min())} or {self.start_PM} or {intg.rewind}")

            t1['next-candidate'], t1['next-m'], t1['next-s'] = t1['best-candidate'], t1['best-m'], t1['best-s']
            p = 'PM'

        elif tX.shape[0] < 20:
            t1['next-candidate'], t1['next-m'], t1['next-s'] = self.next_from_model(next_type='KG')
            p = 'KG'
        else:
            t1['next-candidate'], t1['next-m'], t1['next-s'] = self.next_from_model(next_type='EI')
            p = 'EI'

        intg.candidate |= {'csteps':self._postprocess_ccsteps(list(t1['next-candidate'])[0])}

        t1[     'opt-time'] = perf_counter() - opt_time_start
        t1['cumm-opt-time'] = perf_counter() - opt_time_start + last_opt_time

        t1['cumm-compute-time'] = intg.pseudointegrator._compute_time

        t1['cumm-time'] = (intg.pseudointegrator._compute_time
                           + perf_counter() - opt_time_start + last_opt_time)

        self.pd_opt = pd.concat([self.pd_opt, t1], ignore_index=True)

        with pd.option_context('display.max_rows'         , None, 
                               'display.max_columns'      , None,
                               'display.precision'        , 3,
                               'display.expand_frame_repr', False,
                               'display.max_colwidth'     , 100):
#            print(self.pd_opt[['test-m','best-m','cumm-time','base-time']])
            print(self.pd_opt)
            print("Optimisation type: ", p)
            self.pd_opt.to_csv(self.outf, index=False)

    def store_test_from_model(self, t1):
        test = self.torch.tensor([list(t1['test-candidate'])[0]], **self.torch_kwargs)
        _, t1['check-m'], t1['check-s'] = self.substitute_candidate_in_model(test)
        return t1

    def store_validation_from_model(self, t1,i, test_case):
        test = self.torch.tensor([test_case], **self.torch_kwargs)
        _, t1[f'test-{i}-m'], t1[f'test-{i}-s'], = self.substitute_candidate_in_model(test)
        return t1

    def add_to_model(self, tX, tY, tYv):

        from gpytorch.mlls             import ExactMarginalLogLikelihood
        from botorch.models            import FixedNoiseGP
        from botorch.fit               import fit_gpytorch_model as fit_model
        from botorch.models.transforms import Standardize, Normalize

        self.bounds = tX
        #print(self.bounds)

        self.normalise = Normalize(d=4, bounds=self._bounds)
        self.standardise = Standardize(m=1)

        self.norm_X       = self.normalise.transform(  self.torch.tensor(tX , **self.torch_kwargs))
        stan_Y, stan_Yvar = self.standardise(self.torch.tensor(tY , **self.torch_kwargs),
                                             self.torch.tensor(tYv, **self.torch_kwargs))

        self.model = FixedNoiseGP(train_X = self.norm_X, train_Y = stan_Y, train_Yvar = stan_Yvar)

        mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, model = self.model)

        mll = mll.to(**self.torch_kwargs)                      
        fit_model(mll)

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, tX:np.ndarray):

        # Create happening region 
        # a cube of length 2*radius around the best points
        var_m = 2
        radius = int(np.ceil(np.sqrt(tX.shape[0])))
        best_cands = tX[tX.argsort(axis=0)[:, -2] <= radius, :]
        means, stds = np.mean(best_cands, axis=0), np.std( best_cands, axis=0)
        hr = self.torch.tensor(np.array([means - var_m * stds,
                                means + var_m * stds]), **self.torch_kwargs)

        # Increase bounds to include happening region
        self._bounds[0, self._bounds[0, :] > hr[0, :]] = hr[0, self._bounds[0, :] > hr[0, :]]
        self._bounds[1, self._bounds[1, :] < hr[1, :]] = hr[1, self._bounds[1, :] < hr[1, :]]
        self._bounds[0, self._bounds[0, :] < 0] = 0

    def _preprocess_csteps(self, csteps):
        '''
            Processing shall be done ONLY for:
                V-cycles
                Starting smoothing iteration 1
                Restriction iterations same
                Prolongation iterations same
        '''
        self.depth = (len(csteps)-1)//2 -1
        return csteps[1], csteps[self.depth+1], csteps[-2], csteps[-1]

    def _postprocess_ccsteps(self, ccsteps):
        return (1, *((ccsteps[0],)*self.depth), ccsteps[1], 
                   *((ccsteps[2],)*self.depth), ccsteps[3])

    def best_from_model(self, *, raw_samples=4096, num_restarts=100):

        from botorch.acquisition import PosteriorMean

        _acquisition_function = PosteriorMean(
            self.model, 
            maximize = False,
            )

        X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
        X_b    = self.normalise.untransform(X_cand)

        return self.substitute_candidate_in_model(X_b)

    def next_from_model(self, *, next_type = 'EI', raw_samples=1024, num_restarts=10,):

        if next_type == 'KG':

            from botorch.acquisition           import qKnowledgeGradient
            from botorch.acquisition.objective import ScalarizedPosteriorTransform
            
            self.torch.random.manual_seed(0)
            _acquisition_function = qKnowledgeGradient(
                self.model, 
                posterior_transform = ScalarizedPosteriorTransform(weights=self.torch.tensor([-1.0], **self.torch_kwargs)),
                num_fantasies       = 128,
                )
            X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
            X_b = self.normalise.untransform(X_cand)

        elif next_type == 'EI':

            from botorch.acquisition           import qNoisyExpectedImprovement
            from botorch.acquisition.objective import ScalarizedPosteriorTransform
            
            self.torch.random.manual_seed(0)
            _acquisition_function = qNoisyExpectedImprovement(
                self.model, self.norm_X,
                posterior_transform = ScalarizedPosteriorTransform(weights=self.torch.tensor([-1.0], **self.torch_kwargs)),
                prune_baseline = True,
                )
            X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
            X_b = self.normalise.untransform(X_cand)
        else:
            raise ValueError(f'next_type {next_type} not recognised')

        return self.substitute_candidate_in_model(X_b)

    def optimise(self, _acquisition_function, raw_samples=4096, num_restarts=100):
        """ Returns a transformed attached candidate which minimises the acquisition function
        """

        from botorch.optim.optimize import optimize_acqf

        self.torch.random.manual_seed(0)
        X_cand, _ = optimize_acqf(
            acq_function = _acquisition_function,
            bounds       = self.normalise.transform(self._bounds),
            q            = 1,
            num_restarts = num_restarts,
            raw_samples  = raw_samples,
        )

        return X_cand

    def substitute_candidate_in_model(self, X_b):
        """Accept an untransformed candidate into model. Get an untransformed output.

        Args:
            X_b: Untransformed attached tensor of shape (1,4)

        Returns:
            X_best: tuple[float64, float64, float64, float64]
            Y_mean: float64
            Y_std : float64
        """

        X_best = self.normalise.transform(X_b)

        Y_lower, Y_upper = self.model.posterior(X_best).mvn.confidence_region()

        Y_mean = self.model.posterior(X_best).mvn.mean
        Y_m    = self.standardise.untransform(Y_mean)[0].squeeze().detach().cpu().numpy()
        Y_l    = self.standardise.untransform(Y_lower)[0].squeeze().detach().cpu().numpy()
        Y_u    = self.standardise.untransform(Y_upper)[0].squeeze().detach().cpu().numpy()
        Y_std = (Y_u - Y_l)/4

        return [tuple(X_b.squeeze().detach().cpu().numpy())], Y_m, Y_std
