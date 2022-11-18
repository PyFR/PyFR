from pyfr.plugins.base import BasePlugin

class RewindPlugin(BasePlugin):
    name = 'rewind'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.__save_step   = self.cfg.getfloat(self.cfgsect, 'save-step',   6)
        self.__rewind_step = self.cfg.getfloat(self.cfgsect, 'rewind-step', 14)

        if intg.cfg.get('solver-time-integrator', 'pseudo-controller') == 'local-pi':
            self.rewind_Δτ = True
        else:
            self.rewind_Δτ = False

        # Enable saving of solution
        intg.rewind = False

        # Save matrices for first time
        if self.rewind_Δτ:
            intg.pseudointegrator.pintg.save_Δτ()
        intg.save_soln()

    def __call__(self, intg):

        if intg.nacptsteps == self.__rewind_step:
            intg.rewind = True

        if intg.rewind == True:
            intg.rewind_soln()
            if self.rewind_Δτ:
                intg.pseudointegrator.pintg.rewind_Δτ()
            print(f"rewind over")
            intg.rewind = False

        if intg.nacptsteps == self.__save_step:
            intg.save_soln()
            if self.rewind_Δτ:
                intg.pseudointegrator.pintg.save_Δτ()
            print(f"save")
