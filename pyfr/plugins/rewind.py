from pyfr.plugins.base import BasePlugin

class RewindPlugin(BasePlugin):
    name = 'rewind'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.__save_step   = self.cfg.getfloat(self.cfgsect, 'save-step',   100)
        self.__rewind_step = self.cfg.getfloat(self.cfgsect, 'rewind-step', 300)

        if intg.cfg.get('solver-time-integrator', 'pseudo-controller') == 'local-pi':
            self.rewind_Δτ = True
        else:
            self.rewind_Δτ = False

        if intg.cfg.hasopt('solver-dual-time-integrator-multip', 'cycle'):
            self.rewind_multip_Δτ = True
        else:
            self.rewind_multip_Δτ = False

        # Enable saving of solution
        intg.save   = True
        intg.rewind = False

        self.saved_nacptsteps = intg.nacptsteps
        
        self.rewind_clock = 0
        
        # Save matrices for first time
        if self.rewind_Δτ and self.rewind_multip_Δτ:
            intg.pseudointegrator.pintg.save_Δτ()
        else:
            intg.pseudointegrator.save_Δτ()

        intg.save_soln()

    def __call__(self, intg):

        self.rewind_clock += 1

        if self.rewind_clock % self.__save_step == 0:           # To be removed
            intg.save = True                              

        if self.rewind_clock % self.__rewind_step == 0:           # To be removed
            intg.rewind = True                              

        if intg.rewind == True:
            intg.rewind_soln()
            intg.tcurr      = self.saved_tcurr
            self.nrjctsteps = intg.nacptsteps - self.saved_nacptsteps
            intg.nacptsteps = self.saved_nacptsteps

            if self.rewind_Δτ and self.rewind_multip_Δτ:
                intg.pseudointegrator.pintg.rewind_Δτ()
            else:
                intg.pseudointegrator.rewind_Δτ()

        if intg.save == True:
            intg.save_soln()
            self.saved_tcurr      = intg.tcurr
            self.saved_nacptsteps = intg.nacptsteps
            if self.rewind_Δτ and self.rewind_multip_Δτ:
                intg.pseudointegrator.pintg.save_Δτ()
            else:
                intg.pseudointegrator.save_Δτ()

