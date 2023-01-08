from pyfr.plugins.base import BasePlugin

class RewindPlugin(BasePlugin):
    name = 'rewind'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.rewind_Δτ = intg.cfg.get('solver-time-integrator', 
                                      'pseudo-controller') == 'local-pi'
        if self.rewind_Δτ:
            self.Δτᵢ = intg.cfg.get('solver-time-integrator', 'pseudo-dt')

        self.rewind_multip_Δτ = intg.cfg.hasopt('solver-dual-time-integrator-multip', 'cycle')

        # Enable saving of solution
        intg.save   = True
        intg.rewind = False

        self.saved_nacptsteps = intg.nacptsteps
        
        #self.rewind_clock = 0
        
        # Save matrices for first time

        intg.save_soln()
        self.saved_tcurr      = intg.tcurr
        self.saved_nacptsteps = intg.nacptsteps
        if self.rewind_Δτ and self.rewind_multip_Δτ:
            intg.pseudointegrator.pintg.save_Δτ()
        else:
            intg.pseudointegrator.save_Δτ()

    def __call__(self, intg):

        if intg.rewind:
            intg.rewind_soln()
            intg.tcurr      = self.saved_tcurr
            self.nrjctsteps = intg.nacptsteps - self.saved_nacptsteps
            intg.nacptsteps = self.saved_nacptsteps

            if self.rewind_Δτ and self.rewind_multip_Δτ:
                intg.pseudointegrator.pintg.rewind_Δτ()
            else:
                intg.pseudointegrator.rewind_Δτ()
            print("Rewind complete.")

        if intg.save:
            intg.save_soln()
            self.saved_tcurr      = intg.tcurr
            self.saved_nacptsteps = intg.nacptsteps
            if self.rewind_Δτ and self.rewind_multip_Δτ:
                intg.pseudointegrator.pintg.save_Δτ()
            else:
                intg.pseudointegrator.save_Δτ()
