import itertools as it
import re
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class TavgPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        
        comm, rank, root = get_comm_rank_root()
        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')
        
        # Std deviation mode
        self.std_mode = self.cfg.get(cfgsect, 'std-mode', 'summary')
        if self.std_mode not in {'summary', 'all'}:
            raise ValueError('Invalid standard deviation mode')

        # Expressions pre-processing
        self._prepare_exprs()

        # Floating point precision
        self.Δh = np.finfo(np.float64).eps

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')
        if fpdtype == 'single':
            self.fpdtype = np.float32
        elif fpdtype == 'double':
            self.fpdtype = np.float64
        else:
            raise ValueError('Invalid floating point data type')

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the file writer
        self._writer = NativeWriter(intg, basedir, basename, 'tavg')

        # Gradient pre-processing
        self._init_gradients()

        # Time averaging parameters
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

        # Get total solution points for the region
        em = intg.system.ele_map
        ergn = self._ele_regions
        if self.cfg.get(self.cfgsect, 'region') == '*':
            tpts = sum(em[e].nupts * em[e].neles for _, e, _ in ergn)
        else:
            tpts = sum(len(r) * em[e].nupts for _, e, r in ergn)

        # Reduce
        self.tpts = comm.reduce(tpts, op=mpi.SUM, root=root)

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect
        c = self.cfg.items_as('constants', float)
        self.anames, self.aexprs = [], []
        self.outfields, self.fexprs = [], []
        self.vnames, self.fnames = [], []
        
        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect, prefix='avg-'):
            self.anames.append(k.removeprefix('avg-'))
            self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))
            self.outfields.append(k)

        # Followed by any functional expressions
        for k in cfg.items(cfgsect, prefix='fun-avg-'):
            self.fnames.append(k.removeprefix('fun-avg-'))
            self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))
            self.outfields.append(k)

        # Create fields for std deviations        
        if self.std_mode == 'all':
            for k in cfg.items(cfgsect, prefix='avg-'):
                self.outfields.append(f'std-{k[4:]}')

            for k in cfg.items(cfgsect, prefix='fun-avg-'):
                self.outfields.append(f'fun-std-{k[4:]}')
        
    def _init_gradients(self):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                           for pname in gradpnames]

    def _init_accumex(self, intg):
        comm, rank, root = get_comm_rank_root()
        self.tstart_acc = self.prevt = self.tout_last = intg.tcurr
        self.prevex = self._eval_acc_exprs(intg)
        self.accex  = [np.zeros_like(p, dtype=np.float64) for p in self.prevex]
        self.vaccex = [np.zeros_like(a) for a in self.accex]

        if intg.save == True:
            self.rprevex = [np.zeros_like(a, dtype=np.float64) for a in self.prevex]
            self.raccex  = [np.zeros_like(a, dtype=np.float64) for a in self.prevex]
            self.rvaccex = [np.zeros_like(a, dtype=np.float64) for a in self.prevex]

    def _eval_acc_exprs(self, intg):
        exprs = []

        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Compute the gradients
        if self._gradpinfo:
            grad_soln = intg.grad_soln

        # Iterate over each element type in the simulation
        for idx, etype, rgn in self._ele_regions:
            soln = intg.soln[idx][..., rgn].swapaxes(0, 1)

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))

            # Prepare any required gradients
            if self._gradpinfo:
                grads = np.rollaxis(grad_soln[idx], 2)[..., rgn]

                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grads,
                                                          self.cfg)

                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            # Evaluate the expressions
            exprs.append(np.array([npeval(v, subs) for v in self.aexprs]))

        return exprs

    def _eval_fun_exprs(self, avars):

        # Prepare the substitution dictionary
        subs = dict(zip(self.anames, avars))
    
        # Evaluate the function and return
        return np.array([npeval(v, subs) for v in self.fexprs])

    def _eval_fun_var(self, dev, accex):
        dfexpr, exprs = [], []
        dh, an = np.sqrt(self.Δh), self.anames

        # Iterate over each element type our averaging region
        for av in accex:
            df = []

            # Evaluate the function
            fx = self._eval_fun_exprs(av)
            exprs.append(fx)

            for i in range(len(an)):
                # Calculate step size
                h = dh * np.maximum(abs(av[i]), dh, where=abs(av[i])>dh, out=np.ones_like(av[i]))                
                av[i] += h
                
                # Evaluate function for the step
                fxh = self._eval_fun_exprs(av)
                
                # Calculate derivatives for functional averages
                df.append((fxh - fx) / h)
                av[i] -= h

            # Stack derivatives     
            dfexpr.append(np.array(df))

        # Multiply by variance and take RMS value
        fv = [np.linalg.norm(df * sd[:, None], axis=0) for df, sd in zip(dfexpr, dev)]
        
        return exprs, fv 

    def _acc_avg_var(self, intg, currex):  
        prevex, vaccex, accex = self.prevex, self.vaccex, self.accex  

        # Weights for online variance and average
        Wmp1mpn = intg.tcurr - self.prevt           # Time from last sample
        W1mpn   = intg.tcurr - self.tstart_acc      # Time from accumilation start
        Wp = 2 * (W1mpn - Wmp1mpn) * W1mpn          # 

        # Iterate over element type
        for v, a, p, c in zip(vaccex, accex, prevex, currex):
            ppc = p + c
            # Accumulate average
            a += Wmp1mpn * ppc

            # Accumulate variance          
            v += Wmp1mpn*(c**2 + p**2 - 0.5 * ppc**2) 
            if Wp !=0:
                v +=  (Wmp1mpn / Wp * (a - W1mpn * ppc)**2)

    def rewind(self, intg):
        if intg.rewind == True:
            for v, a, p, rv, ra, rp in zip(self.vaccex, self.accex, self.prevex, self.rvaccex, self.raccex, self.rprevex):
                v.fill(0);  v += rv
                a.fill(0);  a += ra
                p.fill(0);  p += rp

        elif intg.save == True:
            for v, a, p, rv, ra, rp in zip(self.vaccex, self.accex, self.prevex, self.rvaccex, self.raccex, self.rprevex):
                rv.fill(0); rv += v
                ra.fill(0); ra += a
                rp.fill(0); rp += p

        elif intg.save == False and intg.rewind == False:
            pass # print('Do neither save nor rewind this step.')        
        elif intg.save == intg.rewind == None:
            pass # print('Rewinding is not enabled.')        
        else:
            raise ValueError("Something is wrong with the rewind and save flags.")

    def __call__(self, intg):
        # If we are not supposed to be averaging yet then return
        if intg.tcurr < self.tstart:
            return

        # If necessary, run the start-up routines
        if not self._started:
            self._init_accumex(intg)
            self._started = True

        # See if we are due to write and/or accumulate this step
        dowrite = intg.tcurr - self.tout_last >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_acc_exprs(intg)

            # Accumulate them; always do this even when just writing
            self._acc_avg_var(intg, currex)
            
            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            # Rewind the simulation if necessary
            self.rewind(intg)

            if dowrite:
                comm, rank, root = get_comm_rank_root()
                accex, vaccex = self.accex, self.vaccex
                fdev, funex = [], []   
                wts = 2*(intg.tcurr - self.tstart_acc)
   
                # Normalise the accumulated expressions
                tavg = [a / wts for a in accex]

                # Calculate standard deviation
                dev = [np.sqrt(np.abs(v / wts)) for v in vaccex]

                if self.fexprs: 
                    # Evaluate functional expressions and variance
                    funex, fdev = self._eval_fun_var(dev, tavg)
    
                    # Stack the functional expressions
                    tavg = [np.vstack([a, f]) for a, f in zip(tavg, funex)]

                # Maximum and sum of deviations
                maxd = np.zeros(len(self.fnames) + len(self.anames))
                accd = np.zeros(len(self.fnames) + len(self.anames)) 

                for dx, fx in it.zip_longest(dev, fdev):
                    fdv = np.vstack((dx, fx)) if fdev else dx
                    maxd = np.maximum(np.amax(fdv, axis=(1,2)), maxd)
                    accd += fdv.sum((1, 2))

                # Reduce and output if we're the root rank
                if rank != root:
                    comm.Reduce(maxd, None, op=mpi.MAX, root=root)
                    comm.Reduce(accd, None, op=mpi.SUM, root=root)
                else:
                    comm.Reduce(mpi.IN_PLACE, maxd, op=mpi.MAX, root=root)
                    comm.Reduce(mpi.IN_PLACE, accd, op=mpi.SUM, root=root)
                
                # Stack std deviations and functional deviations
                if self.std_mode == 'all' and self.fexprs:
                    tavg = [np.vstack([a, d, df]) for a, d, df in zip(tavg, dev, fdev)]
                elif self.std_mode == 'all':
                    tavg = [np.vstack([a, d]) for a, d in zip(tavg, dev)]

                # Form the output records to be written to disk
                data = dict(self._ele_region_data)

                for (idx, etype, rgn), d in zip(self._ele_regions, tavg):
                    data[etype] = d.swapaxes(0, 1).astype(self.fpdtype)
                
                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields', ','.join(self.outfields))
                stats.set('tavg', 'tstart', self.tstart_acc)
                stats.set('tavg', 'tend', intg.tcurr)

                # Write summarised stats   
                if rank == root:
                    anm, fnm = self.anames, self.fnames

                    # Write std deviations
                    for an, vm, vc in zip(anm, maxd[:len(anm)], accd[:len(anm)]):
                        stats.set('tavg', f'avg-std-{an}', vc/self.tpts)
                        stats.set('tavg', f'max-std-{an}', vm)
                    
                    # Followed by functional deviations
                    for fn, fm, fc in zip(fnm, maxd[len(anm):], accd[len(anm):]):
                        stats.set('tavg', f'fun-avg-std-{fn}', fc/self.tpts)
                        stats.set('tavg', f'fun-max-std-{fn}', fm)
                            
                intg.collect_stats(stats)

                # If we are the root rank then prepare the metadata
                if rank == root:
                    metadata = dict(intg.cfgmeta,
                                    stats=stats.tostr(),
                                    mesh_uuid=intg.mesh_uuid)
                else:
                    metadata = None

                # Write to disk
                solnfname = self._writer.write(data, intg.tcurr, metadata)

                # If a post-action has been registered then invoke it
                self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                        soln=solnfname, t=intg.tcurr)

                # Reset the accumulators
                if self.mode == 'windowed':
                    for a, v in zip(self.accex, self.vaccex):
                        a.fill(0)
                        v.fill(0)

                    self.tstart_acc = intg.tcurr

                self.tout_last = intg.tcurr