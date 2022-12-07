from pyfr.plugins.base import BasePlugin
from pyfr.mpiutil import get_comm_rank_root

class ModifyConfigPlugin(BasePlugin):
    name = 'modify_configuration'
    systems = ['*']
    formulations = ['dual']
    
    """
        Based on the suffix, optimisation will be applied
        1. If suffix is 'offline'
            config-1.ini will be externally created as a copy of current file.
        2. else:
            2. If suffix is 'online'
                config-1.ini will be created within pyfrs file.
            3. If suffix is 'onfline'
                config-1.ini will be created within pyfrs file.

    """
    
    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        if intg.opt_type == 'onfline':
            print("Offline optimisation modification is performed at runtime.")
        elif intg.opt_type == 'online':
            print("Online optimisation modification.")

        intg.candidate = {}

    def __call__(self, intg):

        if intg.reset_opt_stats:
            print("Resetting optimisation stats.")
            match intg.opt_type, intg.bad_sim:
                case 'online', False:
                    intg.save = True
                case _, True:            
                    intg.rewind = True
                case 'onfline', _:
                    intg.rewind = True
                case None, _:
                    print("No optimisation type specified.")
                case _, _:
                    raise ValueError('Not yet implemented.')

            if intg.candidate == {}:
                print("Optimisation is not running.")
                return

            if intg.opt_type in ['onfline', 'online']:
                if intg.candidate.get('csteps'):
                    intg.pseudointegrator.csteps = intg.candidate.get('csteps')
                intg.candidate = {}

            elif intg.opt_type == 'offline':
                # TODO: Create a new config file, say config-1.ini
                # Add the new config file to the list of config files.
                raise ValueError(f"Not implemented yet.")
            else:
                raise ValueError(f"Required: 'onfline', 'online' or 'offline'.",
                                 f"Given suffix: {intg.opt_type}")

        # TODO: Add the config modifications into pyfrs file in the right names.

    def modify_maximum_pseudo_iterations(self, intg):
        if self.cfg.hasopt('solver-dual-time-integrator-multip', 'cycle'):
            self.maxniters = intg.pseudointegrator.pintg.maxniters
        else:
            self.maxniters = intg.pseudointegrator.maxniters

    def add_config_to_prevcfgs(self, intg):
        if intg.opt_type in ['onfline', 'online']:
            intg.prev_cfgs['opt-cfg'] = intg.cfg.tostr()
