from graphlib import TopologicalSorter

from pyfr.plugins.fields.base import BaseFieldProvider
from pyfr.plugins.fields.cf import CfField
from pyfr.plugins.fields.cp import CpField
from pyfr.plugins.fields.isen_mach import IsentropicMachField
from pyfr.plugins.fields.mach import MachField
from pyfr.plugins.fields.mu import MuField
from pyfr.plugins.fields.tau_wall import TauWallField
from pyfr.plugins.fields.vorticity import VorticityField
from pyfr.plugins.fields.yplus import YPlusField
from pyfr.util import subclasses


def get_field_providers(names, ndims, cfg, export_type):
    # Resolve a list of provider names + their declared deps into a topo-
    # sorted list of instantiated providers, ready for FieldRunner to walk.
    available = {c.name: c for c in subclasses(BaseFieldProvider)}

    ts = TopologicalSorter()
    todo, added = list(names), set()

    while todo:
        if (name := todo.pop()) in added:
            continue

        deps = available[name].deps
        ts.add(name, *deps)
        todo.extend(deps)
        added.add(name)

    plugins = [available[n](ndims, cfg, export_type) for n in ts.static_order()]

    # Transformers mutate view.ploc / view.pris in place; run them before
    # producers so producers see the transformed state.
    return sorted(plugins, key=lambda p: p.kind != 'transformer')
