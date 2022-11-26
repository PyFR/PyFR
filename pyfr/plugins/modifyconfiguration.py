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
            print("Offline optimisation modification done at runtime.")
        elif self.suffix == 'online':
            print("Online optimisation modification.")
                
    def __call__(self, intg):

        if intg.candidate is None:
            raise ValueError("Please run the Bayesian Optimisation plugin first.")

        if intg.candidate == {}:
            return

        if self.suffix in ['onfline', 'online']:
            if intg.candidate.get('csteps'):
                intg.pseudointegrator.csteps = intg.candidate.get('csteps')
            intg.candidate = {}
            # TODO: Add the config modifications into pyfrs file in the right names.

        elif self.suffix == 'offline':
            # TODO: Create a new config file, say config-1.ini
            # Add the new config file to the list of config files.
            raise ValueError(f"Not implemented yet.")
        else:
            raise ValueError(f"Required: 'onfline', 'online' or 'offline'. Given suffix: {self.suffix}")


        