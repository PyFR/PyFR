from pyfr.plugins.base import BasePlugin

class RewindPlugin(BasePlugin):
    name = 'rewind'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.rewind_dtau = self.cfg.get('solver-time-integrator', 
                                      'pseudo-controller') == 'local-pi'

        self.if_rewind = self.cfg.getbool(self.cfgsect, 'if-rewind', False)

        if self.rewind_dtau:
            self.Δτᵢ = self.cfg.get('solver-time-integrator', 'pseudo-dt')

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
        #if self.rewind_dtau and self.rewind_multip_Δτ:
        #    intg.pseudointegrator.pintg.save_dtau()
        #else:
        #    intg.pseudointegrator.save_dtau()
        intg.pseudointegrator.save_dtau()

    def __call__(self, intg):

        if intg.rewind:
            intg.rewind_soln()
            intg.tcurr      = self.saved_tcurr
            self.nrjctsteps = intg.nacptsteps - self.saved_nacptsteps
            intg.nacptsteps = self.saved_nacptsteps

            #if self.rewind_dtau and self.rewind_multip_Δτ:
            #    intg.pseudointegrator.pintg.rewind_dtau()
            #else:
            #    intg.pseudointegrator.rewind_dtau()
            
            if not self.if_rewind:
                intg.pseudointegrator.reset_dtau()
                #print("Reset complete.")

            else:
                intg.pseudointegrator.rewind_dtau()
                print("Rewind complete.")

        if intg.save:
            intg.save_soln()
            self.saved_tcurr      = intg.tcurr
            self.saved_nacptsteps = intg.nacptsteps

            if not self.if_rewind:
                intg.pseudointegrator.reset_dtau()
                print("Reset complete.")
            else:
                intg.pseudointegrator.save_dtau()
                print("Save complete.")
                
            #if self.rewind_dtau and self.rewind_multip_Δτ:
            #    intg.pseudointegrator.pintg.save_dtau()
            #else:
            #    intg.pseudointegrator.save_dtau()
