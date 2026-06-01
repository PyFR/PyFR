from pyfr.snapshot.file import FileSnapshot


class SolnSnapshot(FileSnapshot):
    # Instantaneous solver state.  Data is conservative; con_to_pri /
    # grad_con_to_pri are needed to produce primitives.  Field registry
    # uses visvars grouping (inherited from Snapshot default).  Providers
    # (mach, yplus, ...) supported.
    name = 'soln'
    supports_providers = True

    def to_pris(self, interp_data, cfg):
        return list(self.elementscls.con_to_pri(interp_data, cfg))

    def to_grad_pris(self, interp_data, grad_interp, cfg):
        return list(self.elementscls.grad_con_to_pri(interp_data, grad_interp,
                                                     cfg))
