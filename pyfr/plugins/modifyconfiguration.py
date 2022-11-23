from pyfr.plugins.base import BasePlugin

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

        if self.suffix == 'onfline':
            print("Offline optimisation shall be performed without exiting from simulation.")
    
    def __call__(self, intg):

        if intg.candidate is None:
            raise ValueError("Please run the Bayesian Optimisation plugin first.")

        if intg.candidate == {}:
            return

        if self.suffix == 'onfline':
            print("Offline optimisation shall be performed without exiting from simulation.")

        print("csteps was before: ", intg.pseudointegrator.csteps)
        if intg.candidate.get('csteps'):
            intg.pseudointegrator.csteps = intg.candidate.get('csteps')
        print("csteps  is    now: ", intg.pseudointegrator.csteps)
        intg.candidate = {}




        