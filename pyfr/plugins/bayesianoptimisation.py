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

        self.torch_kwargs = {'dtype' : torch.float64, 'device': torch.device("cpu")}    # Stress test with gpu too, compare timings

        self._bounds = self.torch.tensor(bounds_init, **self.torch_kwargs)

        self.pd_opt =  pd.DataFrame(columns = ['test-candidate', 'test-m', 'test-s',
                                               'check-m', 'check-s',
                                               'next-candidate', 'next-m', 'next-s',
                                               'best-candidate', 'best-m', 'best-s',
                                               ])

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

        t1 =  pd.DataFrame({'test-candidate': [self._preprocess_csteps(intg.pseudointegrator.csteps)] , 
                            'test-m'        : [intg.opt_cost_mean], 
                            'test-s'        : [intg.opt_cost_std ]})

        # Process all optimisables
        tX  = np.array(list(self.pd_opt['test-candidate'])+[list(t1['test-candidate'])[0]])
        tY  = np.array(list(self.pd_opt['test-m'        ])+[list(t1['test-m'        ])[0]]).reshape(-1,1)
        tYv = np.array(list(self.pd_opt['test-s'        ])+[list(t1['test-s'        ])[0]])**2
        tYv = tYv.reshape(-1,1)
        self.add_to_model(tX, tY, tYv)

        t1 = self.store_test_from_model(t1)
        #t1 = self.store_validation_from_model(t1, 0, (0, 0, 0, 0))
        #t1 = self.store_validation_from_model(t1, 1, (1, 1, 1, 1))
        #t1 = self.store_validation_from_model(t1, 5, (5, 5, 5, 5))
        
        t1 = self.store_next_from_model(t1)
        t1 = self.store_best_from_model(t1)
        self.pd_opt = pd.concat([self.pd_opt, t1], ignore_index=True)

        with pd.option_context('display.max_rows'         , None, 
                               'display.max_columns'      , None,
                               'display.precision'        , 3,
                               'display.expand_frame_repr', False,
                               'display.max_colwidth'     , 100):
            print(self.pd_opt)
            self.pd_opt.to_csv('bayesian_optimisation.csv', index=False)


        if intg.rewind:
            intg.candidate |= {'csteps':self._postprocess_ccsteps(list(t1['best-candidate'])[0])}
        else:
            intg.candidate |= {'csteps':self._postprocess_ccsteps(list(t1['next-candidate'])[0])}

    def store_test_from_model(self, t1):
        test = self.torch.tensor([list(t1['test-candidate'])[0]], **self.torch_kwargs)
        _, t1['check-m'], t1['check-s'] = self.substitute_candidate_in_model(test)
        return t1

    def store_validation_from_model(self, t1,i, test_case):
        test = self.torch.tensor([test_case], **self.torch_kwargs)
        _, t1[f'test-{i}-m'], t1[f'test-{i}-s'], = self.substitute_candidate_in_model(test)
        return t1

    def store_next_from_model(self, t1):
        t1['next-candidate'], t1['next-m'], t1['next-s'] = self.next_from_model()
        return t1

    def store_best_from_model(self, t1):
        t1['best-candidate'], t1['best-m'], t1['best-s'] = self.best_from_model()
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

        norm_X            = self.normalise.transform(  self.torch.tensor(tX , **self.torch_kwargs))
        stan_Y, stan_Yvar = self.standardise(self.torch.tensor(tY , **self.torch_kwargs),
                                             self.torch.tensor(tYv, **self.torch_kwargs))

        self.model = FixedNoiseGP(train_X = norm_X, train_Y = stan_Y, train_Yvar = stan_Yvar)

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

    def next_from_model(self, *, raw_samples=1024, num_restarts=10, num_fantasies = 128,):

        from botorch.acquisition           import qKnowledgeGradient
        from botorch.acquisition.objective import ScalarizedPosteriorTransform
        
        self.torch.random.manual_seed(0)
        _acquisition_function = qKnowledgeGradient(
            self.model, 
            posterior_transform = ScalarizedPosteriorTransform(weights=self.torch.tensor([-1.0])),
            num_fantasies       = num_fantasies,
            )
        X_cand = self.optimise(_acquisition_function, raw_samples, num_restarts)
        X_b = self.normalise.untransform(X_cand)

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

        Y_mean           = self.model.posterior(X_best).mvn.mean.squeeze().detach().numpy()
        Y_lower, Y_upper = self.model.posterior(X_best).mvn.confidence_region()

        Y_l = self.standardise.untransform(Y_lower)[0].squeeze().detach().numpy()
        Y_u = self.standardise.untransform(Y_upper)[0].squeeze().detach().numpy()

        Y_std = (Y_u - Y_l)/4

        return [tuple(X_b.squeeze().detach().numpy())], self.standardise.untransform(Y_mean)[0], Y_std
