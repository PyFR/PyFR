from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root

class ModifyConfigPlugin(BasePlugin):
    name = 'modify_configuration'
    systems = ['*']
    formulations = ['dual']
    
    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        if intg.opt_type == 'onfline':
            print("Offline optimisation modification is performed at runtime.")
        elif intg.opt_type == 'online':
            print("Online optimisation modification.")

        intg.candidate = {}

        self.depth = self.cfg.getint('solver', 'order', 0)

    def __call__(self, intg):

        if intg.reset_opt_stats:
            if intg.opt_type == 'online' and not intg.bad_sim:
                intg.save = True
            elif intg.bad_sim:
                intg.rewind = True
            elif intg.opt_type == 'onfline':
                if intg.offline_optimisation_complete:
                    print("Offline optimisation is complete.")
                    intg.rewind = False
                else:
                    intg.rewind = True
            elif intg.opt_type is None:
                print("No optimisation type specified.")
            else:
                raise ValueError('Not yet implemented.')

            if intg.candidate == {}:
                print("Optimisation is not running.")
                return

            if intg.opt_type in ['onfline', 'online']:

                csteps = [0 for s in intg.candidate if s.startswith("cstep:") and s[6:].isdigit()]

                dtaufs = [0 for s in intg.candidate if s.startswith("pseudo-dt-fact:") and s[15:].isdigit()]

                for key, value in intg.candidate.items():
                    if key.startswith("cstep:") and key[6:].isdigit():
                        csteps[int(key[6:])] = value
                    elif key.startswith("pseudo-dt-fact:") and key[15:].isdigit():
                        dtaufs[int(key[15:])] = value
                    elif key.startswith("pseudo-dt-max"):
                        intg.pseudointegrator.pintg.Δτᴹ = value
                    elif key.startswith("pseudo-dt-fact"):
                        intg.pseudointegrator.dtauf = value
                    else:
                        raise ValueError(f"Unknown key: {key}")

                if any(d == 0 for d in dtaufs):
                    raise ValueError(f"Ensure all dtaufs are non-zero.")

                if any(c.startswith('cstep:') for c in intg.candidate):
                    intg.pseudointegrator.csteps = self._postprocess_ccsteps(csteps)

                if any(c.startswith('pseudo-dt-fact:') for c in intg.candidate):
                    intg.pseudointegrator.dtaufs = dtaufs

                intg.candidate.clear()

            elif intg.opt_type == 'offline':
                # TODO: Create a new config file, say config-1.ini
                # Add the new config file to the list of config files.
                raise ValueError(f"Not implemented yet.")
            else:
                raise ValueError(f"Required: 'onfline', 'online' or 'offline'.",
                                 f"Given suffix: {intg.opt_type}")

    def add_config_to_prevcfgs(self, intg):
        if intg.opt_type in ['onfline', 'online']:
            intg.prev_cfgs['opt-cfg'] = intg.cfg.tostr()

    def _postprocess_ccsteps(self, ccsteps):
        #                                                    LVL-highest                                              LVL-lowest                                                                                    LVL-highest 
        if   len(ccsteps) == 7:                     return  (ccsteps[0],)      + (ccsteps[1],)                  + (ccsteps[2],) * (self.depth-2) + (ccsteps[3],) + (ccsteps[4],) * (self.depth-2) + (ccsteps[5],) + (ccsteps[6],)
        elif len(ccsteps) == 6 and self.depth == 3: return         (1.0,)      + (ccsteps[0],)                  + (ccsteps[1],)                  + (ccsteps[2],) + (ccsteps[3],)                  + (ccsteps[4],) + (ccsteps[5],)
        elif len(ccsteps) == 5:                     return  (ccsteps[0],)      + (ccsteps[1],) * (self.depth-1) + (ccsteps[2],)                  + (ccsteps[3],) * (self.depth-1)                                 + (ccsteps[4],)
        elif len(ccsteps) == 4:                     return  (1.0,)             + (ccsteps[0],) * (self.depth-1) + (ccsteps[1],) + (ccsteps[2],) * (self.depth-1) + (ccsteps[3],)
        elif len(ccsteps) == 1:                     return  (1.,) * self.depth                                  + (1,)                           + (1.,) * (self.depth-1)                                         + (ccsteps[0],)
        elif len(ccsteps) == 2:                     return  (1.,) * self.depth                                  + (ccsteps[0],) + (1.,) * (self.depth-1)                                                          + (ccsteps[1],)
        else:
            raise ValueError(f"Ensure that the combination of number of csteps: {len(ccsteps)} and depth: {self.depth} is valid.")
